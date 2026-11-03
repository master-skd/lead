"""Cache parity, resumability, conditioning validation, and complete-data gates."""
import json
from pathlib import Path
import numpy as np
import pytest
from lead.tfv6.hidden_state_cache import (initialize_cache, CacheShardWriter,
                                        HiddenStateCache, build_cache_index)


def prepare(tmp_path, shards=1, count=2):
    model=tmp_path/'model';model.mkdir()
    (model/'config.json').write_text(json.dumps({'vlm_config':{'text_config':{'hidden_size':4}}}))
    (model/'model.safetensors').write_bytes(b'original-model')
    manifest=tmp_path/'selected.jsonl'
    entries=[dict(image_key=f'Scenario/Route/rgb/{i:05d}.jpg',navigation='follow_lane',speed=3.+i) for i in range(count)]
    manifest.write_text(''.join(json.dumps(row)+'\n' for row in entries))
    manifest.with_suffix('.meta.json').write_text(json.dumps(dict(fraction=.1,seed=2026,frames=count)))
    root=tmp_path/'cache'
    signature=initialize_cache(root,manifest,model,shards)
    return root,manifest,model,entries,signature


def tokens(length):
    return np.arange(length*4,dtype=np.float16).reshape(length,4),np.arange(length,dtype=np.int8)%4


def test_packed_cache_preserves_exact_tokens_mask_views_and_portable_keys(tmp_path):
    root,_,_,entries,signature=prepare(tmp_path)
    writer=CacheShardWriter(root,0,signature)
    writer.append_batch([(entries[0],*tokens(3)),(entries[1],*tokens(5))]);writer.close()
    report=build_cache_index(root)
    assert report['frames']==2 and report['bytes']==(3+5)*(4*2+1)
    cache=HiddenStateCache(root)
    result=cache.generate_batch(['/other/disk/'+row['image_key'] for row in entries],['follow_lane']*2,[3.,4.])
    assert result['qwen_hidden'].shape==(2,5,4)
    assert result['qwen_token_mask'].sum(1).tolist()==[3,5]
    for i,n in enumerate([3,5]):
        assert np.array_equal(result['qwen_hidden'][i,:n],tokens(n)[0])
        assert np.array_equal(result['qwen_view_ids'][i,:n],tokens(n)[1])
    assert result['qwen_view_ids'][0,3:].tolist()==[-1,-1]
    assert len(cache.keys())==2
    with pytest.raises(ValueError,match='navigation/speed'):
        cache.generate_batch([entries[0]['image_key']],['turn_left'],[3.])
    with pytest.raises(ValueError,match='navigation/speed'):
        cache.generate_batch([entries[0]['image_key']],['follow_lane'],[8.])
    cache.close()


def test_resume_repairs_only_uncommitted_tail_and_rejects_duplicate_generator(tmp_path):
    root,_,_,entries,signature=prepare(tmp_path)
    writer=CacheShardWriter(root,0,signature)
    with pytest.raises(BlockingIOError):
        CacheShardWriter(root,0,signature)
    writer.append(entries[0],*tokens(3));writer.close()
    binary=root/'shard-00000.bin';committed=binary.stat().st_size
    with binary.open('ab') as stream: stream.write(b'uncommitted feature bytes')
    with (root/'shard-00000.jsonl').open('ab') as stream: stream.write(b'{"interrupted')
    writer=CacheShardWriter(root,0,signature)
    assert binary.stat().st_size==committed and set(writer.completed)=={entries[0]['image_key']}
    writer.append(entries[1],*tokens(4));writer.close()
    assert build_cache_index(root)['frames']==2


def test_incomplete_cache_cannot_be_used_as_smaller_training_set(tmp_path):
    root,_,_,entries,signature=prepare(tmp_path,shards=2)
    writer=CacheShardWriter(root,0,signature)
    writer.append(entries[0],*tokens(3));writer.close()
    with pytest.raises(ValueError,match='missing'):
        build_cache_index(root)
    with pytest.raises(ValueError,match='complete index'):
        HiddenStateCache(root)


def test_cache_fingerprint_rejects_changed_model_even_same_file_size(tmp_path):
    root,manifest,model,_,signature=prepare(tmp_path)
    assert initialize_cache(root,root/'manifest.jsonl',model,1)==signature
    (model/'model.safetensors').write_bytes(b'modified-model')
    with pytest.raises(ValueError,match='different'):
        initialize_cache(root,manifest,model,1)


def test_corrupt_feature_payload_is_never_used_silently(tmp_path):
    root,_,_,entries,signature=prepare(tmp_path)
    writer=CacheShardWriter(root,0,signature)
    writer.append_batch([(row,*tokens(3)) for row in entries]);writer.close()
    build_cache_index(root)
    binary=root/'shard-00000.bin'
    with binary.open('r+b') as stream: stream.write(b'XX')
    cache=HiddenStateCache(root)
    try:
        with pytest.raises(ValueError,match='corrupt'):
            cache.generate_batch([entries[0]['image_key']],['follow_lane'],[3.])
    finally:
        cache.close()


def test_cached_training_preflight_needs_no_qwen_environment_or_weights(tmp_path):
    import os
    import shutil
    import subprocess
    import sys
    root,_,_,entries,signature=prepare(tmp_path)
    writer=CacheShardWriter(root,0,signature)
    writer.append_batch([(row,*tokens(3)) for row in entries]);writer.close()
    build_cache_index(root)
    repo=tmp_path/'repo'
    (repo/'scripts/p8').mkdir(parents=True)
    source=Path(__file__).resolve().parents[1]
    shutil.copyfile(source/'scripts/p8/train_p8_hidden.sh',repo/'scripts/p8/train_p8_hidden.sh')
    for name in ('data/carla_leaderboard2/data','data/carla_leaderboard2/buckets','weights'):
        (repo/name).mkdir(parents=True)
    (repo/'weights/model_0030.pth').write_bytes(b'preflight does not load weights')
    (repo/'weights/config.json').write_text('{}')
    env={**os.environ,'PYTHONPATH':str(source),'P8_LEAD_PYTHON':sys.executable,
         'P8_QWEN_PYTHON':'/missing/qwen/python','P8_QWEN_MODEL':'/missing/qwen/model',
         'P8_HIDDEN_CACHE':str(root),'P8_POSTTRAIN':'weights/model_0030.pth',
         'CUDA_VISIBLE_DEVICES':'0','P8_PREFLIGHT_ONLY':'1'}
    result=subprocess.run(['bash','scripts/p8/train_p8_hidden.sh'],cwd=repo,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert 'preflight OK' in result.stdout and 'hidden_dim=4' in result.stdout


def test_dataset_uses_exact_cached_manifest_without_sampling_a_second_tenth(tmp_path):
    from types import SimpleNamespace
    from lead.data_loader.carla_dataset import CARLAData
    root,_,_,entries,signature=prepare(tmp_path)
    writer=CacheShardWriter(root,0,signature)
    writer.append_batch([(row,*tokens(3)) for row in entries]);writer.close()
    build_cache_index(root)
    class Bucket:
        def __len__(self): return len(self.images)
    bucket=Bucket()
    for name in ('bev_3rd_person_images','images_perturbated','semantics','semantics_perturbated',
                 'hdmap','hdmap_perturbated','depth','depth_perturbated','lidars','radars','radars_perturbated',
                 'bboxes','metas','route_dirs','route_indices','sample_start','global_indices'):
        setattr(bucket,name,np.arange(100))
    bucket.images=np.array([f'/new/disk/Scenario/Route/rgb/{i:05d}.jpg'.encode() for i in range(100)])
    dataset=CARLAData.__new__(CARLAData)
    dataset.rank=1;dataset.random=True
    dataset.config=SimpleNamespace(seed=9,use_hidden_intent_planner=True,use_vlm_intent=False,
        use_semantic_intent_planner=False,hidden_intent_frame_fraction=.1,hidden_intent_sample_seed=2026,
        hidden_intent_cache=str(root),carla_num_samples=0)
    dataset.bucket_collection=SimpleNamespace(buckets=[bucket],buckets_mixture_per_epoch=lambda epoch:{0:1.})
    dataset.shuffle(0)
    assert len(dataset)==2
    assert set(map(lambda key:key.decode().split('/')[-1],dataset.images))=={'00000.jpg','00001.jpg'}
    dataset.shuffle(4)
    assert len(dataset)==2


def test_extraction_oom_halves_batch_and_preserves_every_unwritten_frame(tmp_path,monkeypatch):
    import importlib.util
    from types import SimpleNamespace
    from PIL import Image
    import torch
    import lead.inference.qwendrive_hidden as hidden_module
    root,_,model,entries,_=prepare(tmp_path,count=34)
    data=tmp_path/'images'
    for row in entries:
        image=data/row['image_key'];image.parent.mkdir(parents=True,exist_ok=True)
        Image.new('RGB',(1152,384)).save(image)
    attempts=[]
    class Extractor:
        def __init__(self,*args):
            self.model=SimpleNamespace(vlm=SimpleNamespace(dtype=torch.bfloat16))
            self.hidden_dim=4
        def extract(self,images,commands,speeds):
            attempts.append(list(speeds))
            if len(attempts)==1:
                raise torch.cuda.OutOfMemoryError('simulated OOM')
            values=np.array(speeds,dtype=np.float16)[:,None,None]*np.ones((len(images),2,4),np.float16)
            return dict(qwen_hidden=values,qwen_token_mask=np.ones((len(images),2),bool),
                        qwen_view_ids=np.array([[0,3]]*len(images),np.int8)), {'tokens':[2]*len(images)}
    monkeypatch.setattr(hidden_module,'QwenDriveHiddenExtractor',Extractor)
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    path=Path(__file__).resolve().parents[1]/'scripts/p8/cache_qwendrive_hidden.py'
    spec=importlib.util.spec_from_file_location('p8_cache_generator_test',path)
    generator=importlib.util.module_from_spec(spec);spec.loader.exec_module(generator)
    generator.generate(SimpleNamespace(cache_dir=root,model=model,shard=0,batch_size=32,workers=2,
                                       device='cuda:0',data_root=data))
    assert list(map(len,attempts))==[32,16,16,2]
    assert [speed for batch in attempts[1:] for speed in batch]==[row['speed'] for row in entries]
    assert build_cache_index(root)['frames']==34
