"""P8 closed-loop handoff and launcher tests without CARLA/GPU dependencies."""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from threading import Thread
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
import torch
from lead.inference.p8_closed_loop_intent import P8ClosedLoopIntent
from lead.inference.p8_evaluation import validate_config,result_error,archive_invalid_results
from lead.inference.qwendrive_hidden_client import QwenDriveHiddenClient
from lead.inference.qwendrive_hidden_service import serve
from lead.inference.vlm_image_preprocess import carla_bgr_to_vlm_rgb


def config_dict():
    return dict(use_hidden_intent_planner=True,multimodal_planner=True,multimodal_target_speed=True,
                qwendrive_hidden_dim=12,multimodal_planner_k=3)


def test_live_same_frame_socket_and_confidence_pairing_without_ar():
    raw=np.zeros((384,1152,3),np.uint8)
    raw[:,:384]=[20,50,90];raw[:,384:768]=[130,150,170]
    expected=carla_bgr_to_vlm_rgb(raw,30)
    calls=[]
    class Extractor:
        def extract(self,images,commands,speeds):
            assert len(images)==1 and commands==['follow_lane'] and speeds==[-.01]
            np.testing.assert_array_equal(np.asarray(images[0]),expected)
            calls.append(True)
            return dict(qwen_hidden=np.ones((1,4,12),np.float16),qwen_token_mask=np.ones((1,4),bool),
                        qwen_view_ids=np.array([[0,1,2,3]],np.int8)), dict(layer='final_normalized',hidden_dim=12)
    client_socket,server_socket=socket.socketpair()
    client_socket.settimeout(5)
    client=QwenDriveHiddenClient.__new__(QwenDriveHiddenClient)
    client.sock,client.requests=client_socket,0
    thread=Thread(target=serve,args=(Extractor(),server_socket,None));thread.start()
    bridge=P8ClosedLoopIntent(client)
    try:
        tensors=bridge.prepare(raw,[0,0,0,1,0,0],-.01,frame_id=42,timestamp=2.1)
        assert tensors['qwen_hidden'].shape==(1,4,12)
        assert bridge.last_trace['frames']==[dict(frame_id='42',timestamp=2.1,command='follow_lane',speed=-.01)]
        assert calls==[True] and client.requests==1
    finally:
        bridge.close();thread.join(5);server_socket.close()
    routes=torch.tensor([[[[1.,0.]],[[2.,0.]],[[3.,0.]]]])
    speeds=torch.tensor([[4.,8.,12.]])
    logits=torch.randn(1,3,4)
    prediction=SimpleNamespace(pred_route_multimodal=routes,pred_target_speed_multimodal_scalar=speeds,
        pred_target_speed_multimodal_distribution=logits,pred_route_conf=torch.tensor([[.1,.9,.2]]),
        pred_route_selected_idx=torch.tensor([1]),pred_route=routes[:,1],
        pred_target_speed_scalar=speeds[:,1],pred_target_speed_distribution=logits[:,1])
    bridge.validate_prediction(prediction)
    assert bridge.last_trace['route_speed_pair_verified'] is True
    prediction.pred_target_speed_scalar=speeds[:,0]
    with pytest.raises(RuntimeError,match='selected speed'):
        bridge.validate_prediction(prediction)
    prediction.pred_target_speed_scalar=speeds[:,1]
    prediction.pred_route_selected_idx=torch.tensor([0])
    with pytest.raises(RuntimeError,match='confidence argmax'):
        bridge.validate_prediction(prediction)


def test_live_feature_transport_failure_aborts_instead_of_using_empty_tokens():
    class Client:
        def extract_rgb(self,*args,**kwargs): raise ConnectionError('Qwen died')
    bridge=P8ClosedLoopIntent(Client())
    with pytest.raises(ConnectionError,match='Qwen died'):
        bridge.prepare(np.zeros((384,1152,3),np.uint8),[0,0,0,1,0,0],0.,1)


def test_p8_eval_preflight_is_portable_and_exposes_one_checkpoint(tmp_path):
    project=tmp_path/'project'
    (project/'scripts/p8').mkdir(parents=True)
    source=Path(__file__).resolve().parents[1]
    shutil.copyfile(source/'scripts/p8/eval_p8_b2d.sh',project/'scripts/p8/eval_p8_b2d.sh')
    (project/'data/benchmark_routes/bench2drive').mkdir(parents=True)
    carla=project/'carla';carla.mkdir()
    (carla/'CarlaUE4.sh').write_text('#!/bin/sh\nexit 0\n');(carla/'CarlaUE4.sh').chmod(0o755)
    checkpoint=project/'training';checkpoint.mkdir()
    for name in ('model_0018.pth','model_0019.pth'): (checkpoint/name).touch()
    cfg=config_dict();(checkpoint/'config.json').write_text(json.dumps(cfg))
    signature={'contract':'fixture'}
    (checkpoint/'p8_hidden_cache_signature.json').write_text(json.dumps(signature))
    env={**os.environ,'PYTHONPATH':str(source),'P8_B2D_LEAD_PYTHON':sys.executable,
         'P8_B2D_QWEN_PYTHON':sys.executable,'P8_B2D_QWEN_MODEL':str(project),
         'CARLA_ROOT':str(carla),'P8_B2D_PREFLIGHT_ONLY':'1'}
    args=['bash',str(project/'scripts/p8/eval_p8_b2d.sh'),str(checkpoint/'model_0019.pth'),'0','test']
    result=subprocess.run(args,env=env,capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr
    exposed=project/'outputs/p8_b2d_eval_checkpoints/test'
    assert sorted(path.name for path in exposed.glob('*.pth'))==['model.pth']
    assert (exposed/'model.pth').resolve()==checkpoint/'model_0019.pth'
    assert json.loads((exposed/'p8_hidden_cache_signature.json').read_text())==signature
    cfg['joint_scorer_replaces_conf']=True
    (checkpoint/'config.json').write_text(json.dumps(cfg))
    result=subprocess.run(args,env=env,capture_output=True,text=True,timeout=20)
    assert result.returncode!=0 and 'requires a P8' in result.stderr


def test_route_resume_retries_infrastructure_failures_but_keeps_real_driving_failures(tmp_path):
    for name,status in [('1','Perfect'),('2','Failed - Agent got blocked'),('3','Failed - Agent crashed')]:
        folder=tmp_path/name;folder.mkdir()
        (folder/'checkpoint_endpoint.json').write_text(json.dumps({'_checkpoint':{'records':[{'status':status}]}}))
    folder=tmp_path/'4';folder.mkdir();(folder/'checkpoint_endpoint.json').write_text('{broken')
    assert result_error(tmp_path/'2/checkpoint_endpoint.json') is None
    archived=archive_invalid_results(tmp_path)
    assert len(archived)==2
    assert (tmp_path/'1/checkpoint_endpoint.json').exists()
    assert (tmp_path/'2/checkpoint_endpoint.json').exists()
    assert not (tmp_path/'3/checkpoint_endpoint.json').exists()
    assert list((tmp_path/'3').glob('checkpoint_endpoint.invalid_*.json'))


def test_feature_signature_detects_changed_model_or_processor(tmp_path):
    from lead.tfv6.hidden_state_cache import qwen_feature_fingerprint
    from lead.inference.p8_feature_contract import verify_training_features
    model=tmp_path/'model';model.mkdir()
    (model/'config.json').write_text(json.dumps({'vlm_config':{'text_config':{'hidden_size':12}}}))
    (model/'model.safetensors').write_bytes(b'unchanged weights')
    (model/'tokenizer.json').write_text('{}')
    signature=dict(prompt_version='p8-hidden-v1',layer='final_normalized',transport_dtype='float16',
                   compute_dtype='bfloat16',view_order=['front','front_left','front_right','text'],
                   hidden_dim=12,**qwen_feature_fingerprint(model))
    path=tmp_path/'signature.json';path.write_text(json.dumps(signature))
    assert verify_training_features(path,model)['verified']
    (model/'tokenizer.json').write_text('{"modified":true}')
    with pytest.raises(ValueError,match='model_assets'):
        verify_training_features(path,model)


def test_eval_config_rejects_legacy_scorers_and_invalid_dimensions():
    cfg=config_dict();assert validate_config(cfg)==12
    cfg['route_future_safety_gate']=True
    with pytest.raises(ValueError,match='P8'):
        validate_config(cfg)
