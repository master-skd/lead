"""Packed, resumable FP16 token cache. Raw tokens and view IDs; no pooling."""
import hashlib
import fcntl
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import sqlite3
import time
import numpy as np
from lead.tfv6.semantic_intent_cache import image_key, recover_jsonl

SCHEMA = 1


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(16*1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_rows(path):
    seen = set()
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            key = row['image_key']
            if key in seen or image_key(key) != key:
                raise ValueError(f'duplicate or invalid frame key: {key}')
            seen.add(key)
            yield row


def cache_metadata(root):
    return json.loads((Path(root)/'metadata.json').read_text())


def qwen_feature_fingerprint(model):
    """Shared train/evaluation provenance check; never loads model tensors."""
    model=Path(model)
    versions = {}
    for package in ('torch','torchvision','transformers','numpy','pillow','qwen_drive'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    source = importlib.util.find_spec('qwen_drive')
    qwen_sources = {}
    if source is not None and source.origin:
        directory = Path(source.origin).parent
        qwen_sources = {name: file_sha(directory/name) for name in (
            'scene.py','configuration_qwen_drive.py','modeling_qwen_drive.py') if (directory/name).exists()}
    return dict(
        model_config_sha256=file_sha(model/'config.json'),
        model_weights={path.name:file_sha(path) for path in sorted(model.glob('*.safetensors'))},
        runtime=versions, qwen_sources=qwen_sources,
        extractor_sha256=file_sha(Path(__file__).parents[1]/'inference/qwendrive_hidden.py'),
        model_assets={name:file_sha(model/name) for name in (
            'tokenizer.json','tokenizer_config.json','preprocessor_config.json','processor_config.json')
            if (model/name).exists()})


def initialize_cache(root, manifest, model, shards):
    if shards < 1:
        raise ValueError('shards must be positive')
    root, manifest, model = Path(root), Path(manifest), Path(model)
    sidecar = json.loads(manifest.with_suffix('.meta.json').read_text())
    config = json.loads((model/'config.json').read_text())
    # Same fields as the original P8 cache signature; existing caches remain valid.
    signature = dict(schema=SCHEMA, manifest_sha256=file_sha(manifest), num_shards=shards,
        prompt_version='p8-hidden-v1', layer='final_normalized', transport_dtype='float16', compute_dtype='bfloat16',
        view_order=['front','front_left','front_right','text'],
        hidden_dim=config['vlm_config']['text_config']['hidden_size'], sampling=sidecar,
        **qwen_feature_fingerprint(model))
    if not signature['model_weights']:
        raise ValueError('Qwen weights missing')
    frame_count = sum(1 for _ in manifest_rows(manifest))
    if frame_count == 0 or frame_count != sidecar['frames']:
        raise ValueError('empty manifest or incorrect frame count')
    root.mkdir(parents=True, exist_ok=True)
    meta_path = root/'metadata.json'
    if meta_path.exists():
        if cache_metadata(root) != signature or file_sha(root/'manifest.jsonl') != signature['manifest_sha256']:
            raise ValueError('existing hidden cache has different manifest/model/settings; use a new directory')
        return signature
    temporary = root/'manifest.jsonl.tmp'
    temporary.write_bytes(manifest.read_bytes())
    os.replace(temporary, root/'manifest.jsonl')
    (root/'manifest.meta.json').write_text(json.dumps(sidecar,indent=2)+'\n')
    temporary = root/'metadata.json.tmp'
    temporary.write_text(json.dumps(signature, indent=2)+'\n')
    os.replace(temporary, meta_path)
    return signature


def feature_payload(hidden, views):
    hidden = np.asarray(hidden, dtype='<f2')
    views = np.asarray(views, dtype=np.int8)
    if hidden.ndim != 2 or views.shape != hidden.shape[:1] or not len(hidden):
        raise ValueError('cached features must be [L,D] with [L] view IDs')
    if not np.isfinite(hidden).all() or ((views < 0) | (views > 3)).any():
        raise ValueError('invalid feature values or view IDs')
    return hidden.tobytes(order='C') + views.tobytes()


class CacheShardWriter:
    def __init__(self, root, shard, signature):
        self.root, self.shard, self.signature = Path(root), shard, signature
        self.lock = (self.root/f'shard-{shard:05d}.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._open()
        except BaseException:
            self.close()
            raise

    def _open(self):
        signature, shard = self.signature, self.shard
        self.binary_path = self.root/f'shard-{shard:05d}.bin'
        self.records_path = self.root/f'shard-{shard:05d}.jsonl'
        self.completed = {}
        end = 0
        header_seen = False
        for index, row in enumerate(recover_jsonl(self.records_path)):
            if index == 0:
                if row != {'type':'metadata','signature':signature}:
                    raise ValueError('shard signature differs from cache settings')
                header_seen = True
                continue
            if row['offset'] != end or row['image_key'] in self.completed:
                raise ValueError('non-contiguous or duplicate cache shard records')
            if row['nbytes'] != row['tokens']*(signature['hidden_dim']*2+1):
                raise ValueError('invalid cache record length')
            self.completed[row['image_key']] = row
            end += row['nbytes']
        if not header_seen and self.completed:
            raise ValueError('missing shard header')
        self.binary = self.binary_path.open('r+b' if self.binary_path.exists() else 'w+b')
        if self.binary_path.stat().st_size < end:
            self.binary.close()
            raise ValueError('cache binary shorter than committed index')
        # A kill between binary write and JSON commit leaves only an unused tail.
        self.binary.truncate(end)
        self.binary.seek(end)
        self.records = self.records_path.open('a')
        if not header_seen:
            self.records.write(json.dumps({'type':'metadata','signature':signature})+'\n')
            self.records.flush()

    def append(self, entry, hidden, views):
        self.append_batch([(entry,hidden,views)])

    def append_batch(self, samples):
        pending = {}
        for entry, hidden, views in samples:
            if entry['image_key'] in self.completed or entry['image_key'] in pending:
                raise ValueError('frame is already cached')
            if hidden.shape[-1] != self.signature['hidden_dim']:
                raise ValueError('cached hidden dimension mismatch')
            payload = feature_payload(hidden, views)
            row = {key: entry[key] for key in ('image_key','navigation','speed')}
            row.update(offset=self.binary.tell(), nbytes=len(payload), tokens=len(hidden),
                       sha256=hashlib.sha256(payload).hexdigest())
            self.binary.write(payload)
            pending[entry['image_key']] = row
        self.binary.flush()
        os.fsync(self.binary.fileno())
        # One durable commit per microbatch, after feature bytes reach storage.
        self.records.writelines(json.dumps(row)+'\n' for row in pending.values())
        self.records.flush()
        os.fsync(self.records.fileno())
        self.completed.update(pending)

    def close(self):
        for name in ('binary','records','lock'):
            handle = getattr(self,name,None)
            if handle is not None:
                handle.close()


def build_cache_index(root):
    root = Path(root)
    signature = cache_metadata(root)
    if file_sha(root/'manifest.jsonl') != signature['manifest_sha256']:
        raise ValueError('cache manifest changed')
    expected = {row['image_key']:row for row in manifest_rows(root/'manifest.jsonl')}
    temporary = root/'frames.sqlite.tmp'
    temporary.unlink(missing_ok=True)
    db = sqlite3.connect(temporary)
    try:
        db.execute('CREATE TABLE frames (image_key TEXT PRIMARY KEY, shard INTEGER, payload TEXT)')
        db.execute('CREATE TABLE metadata (signature TEXT)')
        db.execute('INSERT INTO metadata VALUES (?)',(json.dumps(signature,sort_keys=True),))
        completed = set()
        for shard in range(signature['num_shards']):
            records_path = root/f'shard-{shard:05d}.jsonl'
            binary = root/f'shard-{shard:05d}.bin'
            if not records_path.exists() or not binary.exists():
                raise ValueError(f'cache shard {shard} missing; resume extraction')
            end = 0
            for index, row in enumerate(recover_jsonl(records_path)):
                if index == 0:
                    if row != {'type':'metadata','signature':signature}:
                        raise ValueError('cache shard signature mismatch')
                    continue
                source = expected.get(row['image_key'])
                if (source is None or row['image_key'] in completed or source['navigation'] != row['navigation']
                        or source['speed'] != row['speed'] or row['offset'] != end
                        or row['nbytes'] != row['tokens']*(signature['hidden_dim']*2+1)):
                    raise ValueError('cache frame/index mismatch')
                end += row['nbytes']
                if end > binary.stat().st_size:
                    raise ValueError('cache shard binary truncated')
                completed.add(row['image_key'])
                db.execute('INSERT INTO frames VALUES (?,?,?)',(row['image_key'],shard,json.dumps(row)))
        if completed != set(expected):
            raise ValueError(f'incomplete cache: {len(completed)}/{len(expected)} frames; resume extraction')
        db.commit()
        db.close()
        os.replace(temporary,root/'frames.sqlite')
    except BaseException:
        db.close()
        temporary.unlink(missing_ok=True)
        raise
    return dict(frames=len(completed), hidden_dim=signature['hidden_dim'],
                bytes=sum(p.stat().st_size for p in root.glob('shard-*.bin')))


class HiddenStateCache:
    def __init__(self, root):
        self.root = Path(root)
        self.signature = cache_metadata(self.root)
        self.hidden_dim = self.signature['hidden_dim']
        if self.signature['schema'] != SCHEMA or file_sha(self.root/'manifest.jsonl') != self.signature['manifest_sha256']:
            raise ValueError('hidden cache schema/manifest mismatch')
        if not (self.root/'frames.sqlite').exists():
            raise ValueError('hidden cache has no complete index; finish extraction and index it first')
        uri = (self.root/'frames.sqlite').resolve().as_uri()+'?mode=ro&immutable=1'
        self.db = sqlite3.connect(uri, uri=True, check_same_thread=False)
        saved = json.loads(self.db.execute('SELECT signature FROM metadata').fetchone()[0])
        if saved != self.signature:
            self.db.close()
            raise ValueError('hidden cache index is stale')
        for shard in range(self.signature['num_shards']):
            if not (self.root/f'shard-{shard:05d}.bin').is_file():
                self.db.close()
                raise ValueError('hidden cache shard file missing; move the entire cache directory')
        self.handles = {}
        self.requests = 0
        self.last_metrics = None

    def keys(self):
        return {row[0] for row in self.db.execute('SELECT image_key FROM frames')}

    def generate_batch(self, paths, commands, speeds):
        if not (len(paths)==len(commands)==len(speeds)) or not paths:
            raise ValueError('invalid cached hidden batch')
        start = time.monotonic()
        rows, view_rows = [], []
        for path, command, speed in zip(paths,commands,speeds,strict=True):
            key = image_key(str(path))
            found = self.db.execute('SELECT shard,payload FROM frames WHERE image_key=?',(key,)).fetchone()
            if found is None:
                raise KeyError(f'hidden cache missing frame {key}')
            shard, entry = found[0], json.loads(found[1])
            if command != entry['navigation'] or not math.isclose(float(speed),entry['speed'],abs_tol=1e-4,rel_tol=1e-5):
                raise ValueError(f'cached navigation/speed mismatch: {key}')
            if shard not in self.handles:
                self.handles[shard] = os.open(self.root/f'shard-{shard:05d}.bin',os.O_RDONLY)
            body = os.pread(self.handles[shard],entry['nbytes'],entry['offset'])
            if len(body)!=entry['nbytes'] or hashlib.sha256(body).hexdigest()!=entry['sha256']:
                raise ValueError(f'truncated/corrupt hidden features: {key}')
            n = entry['tokens']
            boundary = n*self.hidden_dim*2
            rows.append(np.frombuffer(body[:boundary],dtype='<f2').reshape(n,self.hidden_dim))
            view_rows.append(np.frombuffer(body[boundary:],dtype=np.int8))
        max_length = max(map(len,rows))
        hidden = np.zeros((len(rows),max_length,self.hidden_dim),np.float16)
        mask = np.zeros((len(rows),max_length),bool)
        views = np.full((len(rows),max_length),-1,np.int8)
        for i, (row, view) in enumerate(zip(rows,view_rows,strict=True)):
            hidden[i,:len(row)],mask[i,:len(row)],views[i,:len(row)] = row,True,view
        self.last_metrics = dict(backend='hidden_cache',request_id=self.requests,
                                 seconds=time.monotonic()-start,tokens=mask.sum(1).tolist())
        self.requests += 1
        return dict(qwen_hidden=hidden,qwen_token_mask=mask,qwen_view_ids=views)

    def close(self):
        self.db.close()
        for fd in self.handles.values():
            os.close(fd)
