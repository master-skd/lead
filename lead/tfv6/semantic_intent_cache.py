"""Portable per-frame JSON records and a read-only SQLite lookup for P7."""
from __future__ import annotations

import json
import math
import os
import sqlite3
from pathlib import Path


def image_key(path: str) -> str:
    """Ignore machine-specific prefixes; retain scenario/route/rgb/frame.jpg."""
    parts = Path(path).parts
    if len(parts) < 4 or parts[-2] != 'rgb' or '..' in parts[-4:]:
        raise ValueError(f'not a CARLA RGB frame path: {path}')
    return '/'.join(parts[-4:])


def recover_jsonl(path: Path):
    """Read complete records; repair only an interrupted final write on resume."""
    if not path.exists():
        return
    with path.open('r+b') as stream:
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b'\n'):
                stream.truncate(start)
                break
            yield json.loads(line)  # Complete corrupt lines must not be hidden.


class IntentCache:
    """One read-only connection per training rank; no Qwen dependency."""
    def __init__(self, path: str, manifest: str | None = None):
        uri = Path(path).resolve().as_uri() + '?mode=ro'
        self.connection = sqlite3.connect(uri, uri=True)
        version = self.connection.execute(
            "SELECT value FROM metadata WHERE key='schema_version'"
        ).fetchone()
        if version != ('1',):
            raise ValueError('unsupported P7 intent cache schema')
        if manifest:
            import hashlib
            signature = json.loads(self.connection.execute(
                "SELECT value FROM metadata WHERE key='signature'"
            ).fetchone()[0])
            if signature['manifest_sha256'] != hashlib.sha256(Path(manifest).read_bytes()).hexdigest():
                raise ValueError('P7 training manifest differs from cached generation manifest')

    def generate_batch(self, paths, commands, speeds):
        if not (len(paths) == len(commands) == len(speeds)):
            raise ValueError('P7 cache batch lengths differ')
        results = []
        for path, command, speed in zip(paths, commands, speeds, strict=True):
            key = image_key(str(path))
            found = self.connection.execute(
                'SELECT payload FROM intents WHERE image_key=?', (key,),
            ).fetchone()
            if found is None:
                raise KeyError(f'P7 intent cache missing frame: {key}; generate it before training')
            row = json.loads(found[0])
            if row['navigation'] != command or not math.isclose(
                float(row['speed']), float(speed), abs_tol=1e-3, rel_tol=1e-5,
            ):
                raise ValueError(f'P7 cached navigation/speed differs from training: {key}')
            results.append(row)
        return results

    def close(self):
        self.connection.close()


def build_index(manifest: Path, shard_dir: Path, output: Path):
    """Verify complete manifest coverage and publish an atomic indexed JSON cache."""
    from collections import Counter
    from lead.tfv6.semantic_intent_slots import parse_service_result, prepare_planner_intents

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.tmp')
    if temporary.exists():
        temporary.unlink()
    con = sqlite3.connect(temporary)
    signature = None
    histogram, raw_histogram, rejected = Counter(), Counter(), Counter()
    try:
        con.execute('CREATE TABLE intents (image_key TEXT PRIMARY KEY, payload TEXT NOT NULL)')
        con.execute('CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        for shard in sorted(shard_dir.glob('shard-*.jsonl')):
            with shard.open() as stream:
                header = json.loads(next(stream))
                if header.get('type') != 'metadata':
                    raise ValueError(f'missing provenance header: {shard}')
                if signature is None:
                    signature = header['signature']
                if header['signature'] != signature:
                    raise ValueError(f'mixed generation settings: {shard}')
                for line in stream:
                    row = json.loads(line)
                    con.execute('INSERT INTO intents VALUES (?,?) ON CONFLICT(image_key) '
                                'DO UPDATE SET payload=excluded.payload',
                                (row['image_key'], json.dumps(row, ensure_ascii=False)))
            con.commit()
        if signature is None:
            raise ValueError('no generation shards found')
        import hashlib
        if signature['manifest_sha256'] != hashlib.sha256(manifest.read_bytes()).hexdigest():
            raise ValueError('generation manifest fingerprint mismatch')
        count = fallback = 0
        with manifest.open() as stream:
            for line in stream:
                expected = json.loads(line)
                found = con.execute('SELECT payload FROM intents WHERE image_key=?',
                                    (expected['image_key'],)).fetchone()
                if found is None:
                    raise ValueError(f"unfinished generation; missing {expected['image_key']}")
                row = json.loads(found[0])
                if row['navigation'] != expected['navigation'] or row['speed'] != expected['speed']:
                    raise ValueError(f"manifest metadata differs: {row['image_key']}")
                proposal = parse_service_result(row)
                packed = prepare_planner_intents(proposal, row['navigation'])
                valid_count = sum(packed['valid'])
                histogram[valid_count] += 1
                raw = proposal.get('intents') if isinstance(proposal, dict) else None
                raw_histogram[len(raw) if isinstance(raw, list) else 0] += 1
                rejected.update(packed['rejected'])
                fallback += packed['fallback_required']
                count += 1
        summary = {'frames': count, 'fallback': fallback,
                   'valid_branch_histogram_before_fallback': dict(histogram),
                   'raw_branch_histogram': dict(raw_histogram), 'rejections': dict(rejected)}
        con.executemany('INSERT INTO metadata VALUES (?,?)', [
            ('schema_version', '1'), ('signature', json.dumps(signature)),
            ('summary', json.dumps(summary)),
        ])
        con.commit()
        con.close()
        os.replace(temporary, output)
        output.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        return summary
    finally:
        con.close()
        if temporary.exists():
            temporary.unlink()
