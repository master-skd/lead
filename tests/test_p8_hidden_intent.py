"""P8 contracts: attention masks, gradient, migration, same-frame transport."""
from pathlib import Path
import socket
from threading import Thread
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
import torch

from lead.tfv6.hidden_intent_encoder import HiddenIntentEncoder
from lead.inference.qwendrive_hidden import pack_inputs, QwenDriveHiddenExtractor
from lead.inference.qwendrive_hidden_client import QwenDriveHiddenClient
from lead.inference.qwendrive_hidden_service import serve
from lead.data_loader.frame_subset import select_frame_fraction


def test_encoder_masks_padding_and_backpropagates_only_into_trainable_adapter():
    torch.manual_seed(9)
    encoder = HiddenIntentEncoder(12, 16, 3, 4)
    hidden = torch.randn(2, 7, 12, requires_grad=True)
    mask = torch.tensor([[1,1,1,1,0,0,0], [1,1,1,1,1,1,0]], dtype=torch.bool)
    views = torch.full((2,7), 3)
    command = torch.tensor([[0.,0.,0.,1.,0.,0.]]*2)
    speed = torch.tensor([3., 4.])
    result = encoder(hidden, mask, views, command, speed)
    changed = hidden.detach().clone()
    changed[~mask] = float('nan')
    assert torch.allclose(result, encoder(changed, mask, views, command, speed))
    assert result.shape == (2,3,16)
    assert not torch.allclose(result[:,0], result[:,1])
    result[...,0].sum().backward()
    assert hidden.grad is None
    assert encoder.mode_queries.grad.abs().sum() > 0
    assert encoder.adapter[1].weight.grad.abs().sum() > 0
    assert encoder.cross_attention.in_proj_weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match='at least one'):
        encoder(hidden, torch.zeros_like(mask), views, command, speed)


class Processor:
    image_token_id = 77
    merge_size = 2
    tokenizer = SimpleNamespace(pad_token_id=0)
    def encode_vqa(self, images, prompt, device):
        assert len(images) == 3
        # Encode different text lengths to exercise actual batch padding.
        text = [5,6] if '3.000' in prompt else [5,6,7]
        return dict(input_ids=torch.tensor([[1,77,2,77,2,77,2]+text]),
                    image_grid_thw=torch.tensor([[1,2,2]]*3),
                    pixel_values=torch.ones(12,4))


def test_pack_preserves_three_camera_order_mask_and_navigation_text():
    image = Image.new('RGB', (1152,384))
    image.paste((255,0,0),(0,0,384,384))
    image.paste((0,255,0),(384,0,768,384))
    class Checked(Processor):
        def encode_vqa(self, images, *args, **kwargs):
            assert [im.getpixel((0,0)) for im in images] == [(0,255,0),(255,0,0),(0,0,0)]
            return super().encode_vqa(images,*args,**kwargs)
    inputs, mask, view = pack_inputs(Checked(), [image,image], ['follow_lane','turn_left'], [3.,4.])
    assert inputs['input_ids'].shape == (2,10)
    assert mask.sum(1).tolist() == [9,10]
    assert view[0,0] == -1
    assert view[inputs['input_ids']==77].tolist() == [0,1,2,0,1,2]
    assert inputs['image_grid_thw'].shape == (6,3)


def test_extractor_uses_one_base_forward_per_microbatch_no_generate_or_lm_head():
    calls = []
    class Base:
        def __call__(self, **inputs):
            assert inputs['use_cache'] is False and not inputs['output_hidden_states']
            assert 'mm_token_type_ids' in inputs
            calls.append(inputs['input_ids'].shape)
            return SimpleNamespace(last_hidden_state=inputs['input_ids'][...,None].float().expand(-1,-1,12))
    extractor = QwenDriveHiddenExtractor.__new__(QwenDriveHiddenExtractor)
    extractor.model = SimpleNamespace(processor=Processor(), vlm=SimpleNamespace(model=Base()),
                                     _modality_ids=lambda ids: (ids==77).long())
    extractor.device, extractor.microbatch, extractor.hidden_dim = torch.device('cpu'), 2, 12
    result, metadata = extractor.extract([Image.new('RGB',(1152,384))]*3, ['follow_lane']*3,[3.,4.,3.])
    assert calls == [torch.Size([2,10]),torch.Size([1,9])]
    assert result['qwen_hidden'].shape == (3,10,12)
    assert result['qwen_token_mask'].sum(1).tolist() == [9,10,9]
    assert result['qwen_view_ids'][0,-1] == -1
    assert metadata['layer'] == 'final_normalized'


def test_fixed_tenth_sampling_is_epoch_order_and_machine_independent():
    images = np.array([f'/disk1/Scenario/Route/rgb/{i:05}.jpg'.encode() for i in range(100)])
    first = select_frame_fraction(images, np.arange(100), .1, 2026)
    second = select_frame_fraction(images, np.arange(99,-1,-1), .1, 2026)
    moved = np.array([im.replace(b'disk1',b'disk2') for im in images])
    assert len(first) == 10 and set(first)==set(second)
    assert set(first)==set(select_frame_fraction(moved,np.arange(100),.1,2026))
    assert set(first)!=set(select_frame_fraction(images,np.arange(100),.1,2027))


def test_binary_service_preserves_same_frame_features_and_conditions(tmp_path):
    path = tmp_path/'frame.jpg'
    Image.new('RGB',(1152,384)).save(path)
    class Extractor:
        def extract(self, strips, commands, speeds):
            assert commands==['follow_lane'] and speeds==[3.]
            assert strips[0].size==(1152,384)
            return dict(qwen_hidden=np.ones((1,4,12),dtype=np.float16),
                        qwen_token_mask=np.ones((1,4),dtype=bool),
                        qwen_view_ids=np.array([[0,1,2,3]],dtype=np.int8)), {'layer':'final_normalized'}
    client_sock, server_sock = socket.socketpair()
    client_sock.settimeout(5)
    client = QwenDriveHiddenClient.__new__(QwenDriveHiddenClient)
    client.sock, client.requests = client_sock, 0
    thread = Thread(target=serve, args=(Extractor(),server_sock,tmp_path))
    thread.start()
    try:
        arrays = client.generate_batch([path],['follow_lane'],[3.])
        assert arrays['qwen_hidden'].shape==(1,4,12)
        assert client.last_metrics['frames']==[dict(frame_id=str(path),command='follow_lane',speed=3.,timestamp=None)]
        assert client.requests==1
    finally:
        client.close();thread.join(5);server_sock.close()
    assert not thread.is_alive()


def test_client_rejects_stale_or_reordered_frame():
    from lead.inference.qwendrive_intent_client import _receive_json, _send_json
    client_sock, server_sock = socket.socketpair()
    client_sock.settimeout(5)
    client = QwenDriveHiddenClient.__new__(QwenDriveHiddenClient)
    client.sock, client.requests = client_sock, 2
    def worker():
        _receive_json(server_sock)
        _send_json(server_sock,dict(request_id=1,frames=[]))
    thread=Thread(target=worker);thread.start()
    try:
        with pytest.raises(ValueError,match='alignment'):
            client.generate_batch(['current.jpg'],['follow_lane'],[3.])
    finally:
        client.close();thread.join(5);server_sock.close()


def config():
    from lead.training.config_training import TrainingConfig
    cfg=TrainingConfig(loaded_config={'radar_detection':False,'use_radar_detection':False})
    cfg.use_mixed_precision_training=False
    cfg.transfuser_token_dim=32
    cfg.transfuser_num_bev_cross_attention_heads=4
    cfg.transfuser_num_bev_cross_attention_layers=1
    cfg.num_route_points_prediction=3
    cfg.multimodal_planner=True
    cfg.multimodal_planner_k=3
    cfg.multimodal_target_speed=True
    cfg.joint_scorer_replaces_conf=False
    cfg.use_hidden_intent_planner=True
    cfg.qwendrive_hidden_dim=12
    cfg.use_control_conditioning=False
    cfg.use_radars=False
    cfg.use_collision_cost=False
    cfg.route_select_by_conf=True
    return cfg


def test_planner_all_latent_modes_wta_speed_and_posttrain_query_migration():
    from lead.tfv6.planning_decoder import PlanningDecoder
    from lead.training.training_utils import _migrate_planner_query
    from lead.tfv6.joint_online_training import select_route_speed_pair
    cfg=config()
    decoder=PlanningDecoder(8,cfg,torch.device('cpu'))
    old=torch.arange((3+cfg.num_way_points_prediction+1)*32).reshape(1,-1,32).float()
    model=torch.nn.Module()
    model.add_module('planning_decoder',decoder)
    migrated=_migrate_planner_query({'planning_decoder.query':old},model,cfg)['planning_decoder.query']
    for k in range(3):
        assert torch.equal(migrated[:,k*3:(k+1)*3],old[:,:3])
    assert torch.equal(migrated[:,9:],old[:,3:])
    data=dict(qwen_hidden=torch.randn(2,9,12),qwen_token_mask=torch.ones(2,9,dtype=torch.bool),
              qwen_view_ids=torch.tensor([[0,1,2,3,3,3,3,3,3]]*2),
              command=torch.tensor([[0.,0.,0.,1.,0.,0.]]*2),speed=torch.tensor([3.,4.]),
              target_point=torch.zeros(2,2),target_point_previous=torch.zeros(2,2),target_point_next=torch.zeros(2,2))
    log={}
    decoder.train();decoder(torch.randn(2,8,4,4),None,None,data,log)
    assert data['route_multimodal'].shape==(2,3,3,2)
    assert data['semantic_intent_valid'].all()
    losses={}
    gt=torch.zeros(2,3,2)
    winner=decoder._route_winner(gt,data)
    decoder._multimodal_route_loss(gt,data,losses,log)
    assert 'loss_route_anchor' not in losses and 'loss_route_corridor' not in losses
    speed=data['target_speed_multimodal_distribution'][torch.arange(2),winner]
    (sum(losses.values())+speed.square().mean()).backward()
    assert decoder.hidden_intent_encoder.mode_queries.grad.abs().sum()>0
    assert decoder.semantic_query_gate_logit.grad is not None
    assert 'hidden_intent_residual_ratio' in log
    assert 'hidden_route_pair_distance' in log
    pick=data['route_conf'].argmax(1)
    route,_,scalar=select_route_speed_pair(data['route_multimodal'],data['target_speed_multimodal_distribution'],
                                         data['target_speed_multimodal_scalar'],pick)
    assert torch.equal(route,data['route_multimodal'][torch.arange(2),pick])
    assert torch.equal(scalar,data['target_speed_multimodal_scalar'][torch.arange(2),pick])


def test_hidden_mode_refuses_ar_mode_and_wrong_posttrain_length():
    from lead.tfv6.planning_decoder import PlanningDecoder
    from lead.training.training_utils import _migrate_planner_query
    cfg=config();cfg.use_semantic_intent_planner=True
    with pytest.raises(ValueError,match='mutually exclusive'):
        PlanningDecoder(8,cfg,torch.device('cpu'))
    cfg.use_semantic_intent_planner=False
    decoder=PlanningDecoder(8,cfg,torch.device('cpu'))
    model=torch.nn.Module()
    model.add_module('planning_decoder',decoder)
    wrong=torch.zeros(1,4+cfg.num_way_points_prediction+1,32)
    with pytest.raises(ValueError,match='matching posttrain'):
        _migrate_planner_query({'planning_decoder.query':wrong},model,cfg)


def test_dataset_reuses_same_tenth_even_with_epoch_bucket_sampling(tmp_path):
    from lead.data_loader.carla_dataset import CARLAData
    class Bucket:
        def __len__(self):
            return len(self.images)
    bucket=Bucket()
    names=('bev_3rd_person_images','images','images_perturbated','semantics','semantics_perturbated',
           'hdmap','hdmap_perturbated','depth','depth_perturbated','lidars','radars','radars_perturbated',
           'bboxes','metas','route_dirs','route_indices','sample_start','global_indices')
    for name in names:
        setattr(bucket,name,np.arange(100))
    bucket.images=np.array([f'/data/Scenario/Route/rgb/{i:05}.jpg'.encode() for i in range(100)])
    dataset=CARLAData.__new__(CARLAData)
    dataset.rank=0;dataset.random=True
    dataset.config=SimpleNamespace(seed=9,use_hidden_intent_planner=True,use_vlm_intent=False,
        use_semantic_intent_planner=False,hidden_intent_frame_fraction=.1,hidden_intent_sample_seed=2026,
        carla_num_samples=0,logdir=str(tmp_path))
    calls=[]
    def mixture(epoch):
        calls.append(epoch)
        return {0:.25 if epoch==0 else .75}
    dataset.bucket_collection=SimpleNamespace(buckets=[bucket],buckets_mixture_per_epoch=mixture)
    dataset.shuffle(0)
    first=set(dataset.images)
    dataset.shuffle(11)
    assert len(dataset)==10 and set(dataset.images)==first
    assert calls==[0,0]
    assert len((tmp_path/'p8_train_frames.txt').read_text().splitlines())==10


def test_latent_mode_loss_weights_disable_legacy_geometry_losses():
    cfg=config()
    cfg.route_anchor_loss_weight=2
    cfg.use_route_corridor_loss=True
    from lead.common.constants import SourceDataset
    weights=cfg.detailed_loss_weights(int(SourceDataset.CARLA), 0)
    assert weights['loss_route_anchor']==0 and weights['loss_route_corridor']==0
    assert weights['loss_route_conf']>0


def test_p8_live_bridge_transports_current_frame_with_matching_image_processing():
    from lead.inference.p8_closed_loop_intent import P8ClosedLoopIntent
    from lead.inference.vlm_image_preprocess import carla_bgr_to_vlm_rgb
    bgr=np.random.default_rng(4).integers(0,256,(384,1152,3),dtype=np.uint8)
    class Client:
        last_metrics={'request_id':0,'seconds':.1,'frames':[{'frame_id':'42','timestamp':2.1}]}
        def extract_rgb(self,rgb,command,speed,frame_id, timestamp=None):
            assert np.array_equal(rgb,carla_bgr_to_vlm_rgb(bgr,30))
            assert command=='follow_lane' and speed==3. and frame_id==42 and timestamp==2.1
            return dict(qwen_hidden=np.ones((1,4,12),np.float16),qwen_token_mask=np.ones((1,4),bool),
                        qwen_view_ids=np.array([[0,1,2,3]],dtype=np.int8))
    bridge=P8ClosedLoopIntent(Client())
    result=bridge.prepare(bgr,[0,0,0,1,0,0],3.,42,timestamp=2.1)
    assert result['qwen_hidden'].shape==(1,4,12)
    assert bridge.last_trace['frame_id']=='42'


@pytest.mark.parametrize('speed', [-0.0002081831632630484, -0.10249957940203824, -1., 0., 3.])
def test_hidden_prompt_accepts_finite_signed_carla_speed(speed):
    from lead.inference.qwendrive_hidden import hidden_prompt
    assert f'Ego speed: {speed:.3f} m/s.' in hidden_prompt('follow_lane', speed)


@pytest.mark.parametrize('speed', [float('nan'), float('inf'), -float('inf')])
def test_hidden_prompt_rejects_nonfinite_speed(speed):
    from lead.inference.qwendrive_hidden import hidden_prompt
    with pytest.raises(ValueError, match='non-finite'):
        hidden_prompt('follow_lane', speed)


def test_negative_speed_still_reaches_official_processor():
    image = Image.new('RGB', (1152,384))
    inputs, mask, views = pack_inputs(Processor(), [image], ['follow_lane'], [-0.1025])
    assert mask.any() and inputs['image_grid_thw'].shape == (3,3)
