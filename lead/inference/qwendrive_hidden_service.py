"""Serve batched single-forward Qwen-Drive features; feature failures are fatal."""
import argparse
import base64
from io import BytesIO
import json
from pathlib import Path
import socket
import time
from PIL import Image
from lead.inference.qwendrive_intent_client import _receive_json
from lead.inference.qwendrive_hidden_client import send_features


def serve(extractor, connection, media_root):
    while True:
        try:
            request = _receive_json(connection)
        except ConnectionError:
            return
        samples = request['samples']
        if not 0 < len(samples) <= 256:
            raise ValueError('invalid feature request batch size')
        strips = []
        for item in samples:
            if 'image_png' in item:
                with Image.open(BytesIO(base64.b64decode(item['image_png'], validate=True))) as image:
                    strips.append(image.convert('RGB'))
            else:
                path = Path(item['image']).resolve(strict=True)
                if media_root is None or not path.is_relative_to(media_root):
                    raise ValueError('hidden image outside media root')
                with Image.open(path) as image:
                    strips.append(image.convert('RGB'))
        start = time.monotonic()
        arrays, metrics = extractor.extract(strips, [row['command'] for row in samples], [row['speed'] for row in samples])
        meta = dict(request_id=request['request_id'],
                    frames=[{key: row[key] for key in ('frame_id', 'command', 'speed', 'timestamp')} for row in samples],
                    seconds=time.monotonic() - start, **metrics)
        send_features(connection, arrays, meta)
        print(json.dumps({'request_id': request['request_id'], 'batch': len(samples), **metrics}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--socket', required=True)
    ap.add_argument('--media-root')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--microbatch', type=int, default=8)
    ap.add_argument('--expected-hidden-dim', type=int)
    ap.add_argument('--training-signature')
    args = ap.parse_args()
    from lead.inference.qwendrive_hidden import QwenDriveHiddenExtractor
    if args.training_signature:
        from lead.inference.p8_feature_contract import verify_training_features
        print('[qwendrive-hidden] feature contract: '+json.dumps(
            verify_training_features(args.training_signature,args.model)),flush=True)
    extractor = QwenDriveHiddenExtractor(args.model, args.device, args.microbatch)
    if args.expected_hidden_dim is not None and extractor.hidden_dim!=args.expected_hidden_dim:
        raise ValueError(f'Qwen hidden dimension {extractor.hidden_dim} differs from planner {args.expected_hidden_dim}')
    path = Path(args.socket)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    root = Path(args.media_root).resolve(strict=True) if args.media_root else None
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(path))
        server.listen(1)
        print(f'[qwendrive-hidden] ready {path}; hidden_dim={extractor.hidden_dim}; forward only', flush=True)
        try:
            while True:
                connection, _ = server.accept()
                with connection:
                    serve(extractor, connection, root)
        finally:
            path.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
