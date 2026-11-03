"""Preflight and result validation for the P8 confidence-selected closed loop."""
import json
from datetime import datetime,timezone
from pathlib import Path


def validate_config(config):
    required={'use_hidden_intent_planner':True,'use_semantic_intent_planner':False,
              'multimodal_planner':True,'multimodal_target_speed':True,
              'joint_scorer_replaces_conf':False,'use_vlm_intent':False,
              'use_intent_decoder':False,'use_control_conditioning':False,
              'route_speed_safety_gate':False,'route_future_safety_gate':False,
              'route_velocity_scorer_gate':False,'route_predicted_actor_velocity_gate':False}
    bad={key:(config.get(key,False),value) for key,value in required.items()
         if config.get(key,False) is not value}
    if bad:
        raise ValueError(f'This evaluator requires a P8 hidden K-route/K-speed checkpoint: {bad}')
    dimension=config.get('qwendrive_hidden_dim')
    if not isinstance(dimension,int) or dimension<1 or config.get('multimodal_planner_k',0)<1:
        raise ValueError('P8 checkpoint requires valid hidden dimension and mode count')
    return dimension


def result_error(path):
    try:
        data=json.loads(Path(path).read_text())
        records=data['_checkpoint']['records']
        if not records:
            return 'no route records'
        if data.get('entry_status') in ('Crashed','Rejected','Invalid','Started'):
            return f"unfinished entry: {data['entry_status']}"
        for row in records:
            status=row.get('status','')
            if (status not in ('Perfect','Completed','Failed') and not status.startswith('Failed - ')) or any(message in status for message in (
                    'Agent crashed','Simulation crashed',"Agent couldn't be set up",'sensors were invalid')):
                return f'infrastructure failure: {status}'
    except (OSError,ValueError,KeyError,TypeError) as exc:
        return str(exc)
    return None


def archive_invalid_results(root):
    archived=[]
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    for path in Path(root).glob('*/checkpoint_endpoint.json'):
        reason=result_error(path)
        if reason:
            destination=path.with_name(f'checkpoint_endpoint.invalid_{stamp}.json')
            path.rename(destination)
            archived.append(dict(path=str(path),archive=str(destination),reason=reason))
    return archived
