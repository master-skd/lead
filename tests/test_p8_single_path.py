"""Single-route ablation preserves LEAD layout, heads and direct losses."""
from types import SimpleNamespace
import torch
import pytest
from lead.training.config_training import TrainingConfig
from lead.tfv6.planning_decoder import PlanningDecoder
from lead.inference.p8_closed_loop_intent import P8ClosedLoopIntent
from lead.inference.p8_evaluation import validate_config


def config(hidden):
    cfg=TrainingConfig(loaded_config={'radar_detection':False,'use_radar_detection':False})
    cfg.use_mixed_precision_training=False
    cfg.transfuser_token_dim=32
    cfg.transfuser_num_bev_cross_attention_heads=4
    cfg.transfuser_num_bev_cross_attention_layers=1
    cfg.num_route_points_prediction=3
    cfg.multimodal_planner=False
    cfg.multimodal_planner_k=1
    cfg.multimodal_target_speed=False
    cfg.joint_scorer_replaces_conf=False
    cfg.route_select_by_conf=False
    cfg.use_hidden_intent_planner=hidden
    cfg.hidden_intent_single_path=hidden
    cfg.qwendrive_hidden_dim=12
    cfg.use_control_conditioning=False
    cfg.use_radars=False
    cfg.use_collision_cost=False
    return cfg


def data(cfg):
    return dict(qwen_hidden=torch.randn(2,9,12,requires_grad=True),qwen_token_mask=torch.ones(2,9,dtype=torch.bool),
        qwen_view_ids=torch.tensor([[0,1,2,3,3,3,3,3,3]]*2),
        command=torch.tensor([[0.,0.,0.,1.,0.,0.]]*2),speed=torch.tensor([3.,4.]),
        target_point=torch.zeros(2,2),target_point_previous=torch.zeros(2,2),target_point_next=torch.zeros(2,2),
        future_waypoints=torch.zeros(2,cfg.num_way_points_prediction,2),route=torch.zeros(2,3,2),
        target_speed=torch.tensor([3.,4.]),brake=torch.zeros(2,dtype=torch.bool))


def test_single_layout_and_zero_gate_match_original_lead_exactly():
    baseline=PlanningDecoder(8,config(False),torch.device('cpu')).eval()
    cfg=config(True)
    conditioned=PlanningDecoder(8,cfg,torch.device('cpu')).eval()
    result=conditioned.load_state_dict(baseline.state_dict(),strict=False)
    assert not result.unexpected_keys
    assert all(key.startswith(('hidden_intent_encoder.','semantic_query_projector.','semantic_query_gate_logit'))
               for key in result.missing_keys)
    assert conditioned.query.shape==baseline.query.shape==(1,3+cfg.num_way_points_prediction+1,32)
    assert torch.equal(conditioned.query,baseline.query)
    assert not hasattr(conditioned,'conf_decoder') and not hasattr(conditioned,'anchor_embed')
    assert conditioned.hidden_intent_encoder.mode_queries.shape==(1,32)
    with torch.no_grad():conditioned.semantic_query_gate_logit.fill_(-float('inf'))
    sample=data(cfg);bev=torch.randn(2,8,4,4)
    original=baseline(bev,None,None,dict(sample),{})
    output=conditioned(bev,None,None,dict(sample),{})
    for a,b in zip(original,output):
        assert (a is None and b is None) or torch.equal(a,b)


def test_single_route_and_global_speed_use_direct_expert_losses_and_train_adapter():
    cfg=config(True)
    decoder=PlanningDecoder(8,cfg,torch.device('cpu')).train()
    sample=data(cfg)
    outputs=decoder(torch.randn(2,8,4,4),None,None,sample,{})
    route,waypoints,logits,speed,heading=outputs
    assert route.shape==(2,3,2) and logits.shape==(2,len(cfg.target_speed_classes))
    assert not any(key in sample for key in ('route_multimodal','route_conf','target_speed_multimodal_distribution'))
    def fail_wta(*args): raise AssertionError('WTA must not run for the original single-route head')
    decoder._route_winner=fail_wta
    prediction=SimpleNamespace(pred_route=route,pred_future_waypoints=waypoints,
                               pred_target_speed_distribution=logits,pred_target_speed_scalar=speed,pred_headings=heading)
    losses={};decoder.compute_loss(prediction,sample,losses,{})
    assert set(losses)=={'loss_spatial_route','loss_spatio_temporal_waypoints','loss_target_speed'}
    sum(losses.values()).backward()
    assert sample['qwen_hidden'].grad is None
    assert decoder.hidden_intent_encoder.adapter[1].weight.grad.abs().sum()>0
    assert decoder.semantic_query_gate_logit.grad.abs().sum()>0
    assert decoder.target_speed_decoder[0].weight.grad.abs().sum()>0


def test_single_config_and_live_validation_need_no_mode_selection():
    cfg={'use_hidden_intent_planner':True,'hidden_intent_single_path':True,
         'multimodal_planner':False,'multimodal_target_speed':False,
         'multimodal_planner_k':1,'qwendrive_hidden_dim':12}
    assert validate_config(cfg)==12
    bridge=P8ClosedLoopIntent(None,single_path=True)
    bridge.last_trace={}
    prediction=SimpleNamespace(pred_route=torch.zeros(1,3,2),pred_target_speed_scalar=torch.ones(1),
                               pred_target_speed_distribution=torch.ones(1,4))
    bridge.validate_prediction(prediction)
    assert bridge.last_trace['selection']=='single_route'
    prediction.pred_route_conf=torch.ones(1,1)
    with pytest.raises(RuntimeError,match='multimodal'):
        bridge.validate_prediction(prediction)
    wrong=config(True);wrong.multimodal_planner=True;wrong.multimodal_target_speed=True
    with pytest.raises(ValueError,match='original single route/speed'):
        PlanningDecoder(8,wrong,torch.device('cpu'))
