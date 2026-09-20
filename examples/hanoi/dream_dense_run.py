"""Regenerate a Cosmos dense checkpoint's dreams for a recorded deployment run.

For every saved inference input of an OpenPI dense run (``<run>/inference_inputs/*.npz``) the
policy is re-run with the future-frame decode enabled. The dream is the frame the model expects
1.6 s ahead (the end of its 16-row chunk). Outputs, under ``--output``:

* ``dreams.mp4``          [live input | dreamed 1.6 s ahead] per inference, at the run's inference rate
* ``dream_strips/NNNNNN.png``  [live now | dream | live 1.6 s later | difference]
* ``dreams_contact.png``  every ``--contact-every``-th pair on one sheet
* ``dream_report.json``   per inference: offline vs live first row (mm), value, image metrics

Run in the Cosmos environment from the repository root; the checkpoint defaults to the served export.
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from dream_local import l1, label_strip, patch_rope, psnr

DEFAULT_RUN = Path('data/hanoi_cosmos/runs/cosmos_policy/hanoi/hanoi_cosmos_dense_20260919_video_init_cycle2')
AHEAD_S = 1.6


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_RUN / 'exports/iter_000016000.pt')
    parser.add_argument('--stats', type=Path, default=Path('data/hanoi_cosmos/dense_v5/dataset_statistics.json'))
    parser.add_argument('--embeddings', default='data/hanoi_cosmos/t5_embeddings.pkl')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--denoising-steps', type=int, default=5)
    parser.add_argument('--contact-every', type=int, default=8)
    parser.add_argument('--rope', choices=['auto', 'fused', 'unfused'], default='auto')
    args = parser.parse_args()

    run = args.run_dir
    horizon = int(json.loads((args.checkpoint.resolve().parent.parent / 'joint_contract.json').read_text()).get('horizon', 16))
    os.environ['COSMOS_POLICY_PLATFORM'] = 'hanoi_dense'
    os.environ['HANOI_DENSE_HORIZON'] = str(horizon)
    os.environ.setdefault('IMAGINAIRE_OUTPUT_ROOT', str(Path('data/hanoi_cosmos/runs').resolve()))
    import torch
    from PIL import Image, ImageDraw

    from cosmos_policy.datasets.hanoi_joint_data import PROMPT
    from cosmos_policy.experiments.robot.cosmos_utils import get_action
    from cosmos_policy.experiments.robot.hanoi.dense_policy import HanoiDenseInferenceConfig, load_dense_policy, threshold_jaw
    from cosmos_policy.experiments.robot.hanoi.joint_policy import make_joint_observation

    requests = {}
    for line in (run / 'inferences.jsonl').read_text().splitlines():
        event = json.loads(line)
        if event['event'] == 'inference_request':
            requests[int(event['request_id'])] = event['observation_captured_at_s']
    live = {}
    for line in (run / 'events.jsonl').read_text().splitlines():
        event = json.loads(line)
        if event['event'] == 'prediction':
            live[int(event['request_id'])] = np.asarray(event['actions'], np.float64)
    inputs = sorted((run / 'inference_inputs').glob('*.npz'), key=lambda p: int(p.stem))
    ids = [int(p.stem) for p in inputs]
    stamps = np.array([requests.get(i, np.nan) for i in ids])
    frames = {i: np.load(p)['observation/image'] for i, p in zip(ids, inputs)}
    gaps = np.diff(stamps[np.isfinite(stamps)])
    rate = 1 / float(np.median(gaps)) if len(gaps) else 2.0

    patch_rope(args.rope)
    cfg = HanoiDenseInferenceConfig(str(args.checkpoint), str(args.stats), args.embeddings,
                                    num_denoising_steps_action=args.denoising_steps, chunk_size=horizon)
    started = time.perf_counter()
    model, stats, _, identity = load_dense_policy(cfg)
    torch.cuda.synchronize()
    print(f'Loaded {args.checkpoint} ({horizon}-step chunks) in {time.perf_counter() - started:.1f}s', flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    strips_dir = args.output / 'dream_strips'
    strips_dir.mkdir(exist_ok=True)
    encoder = subprocess.Popen(
        ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '696x366',
         '-r', f'{rate:.4f}', '-i', '-', '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
         '-movflags', '+faststart', str(args.output / 'dreams.mp4')], stdin=subprocess.PIPE)
    report = {'run_dir': str(run.resolve()), 'checkpoint': str(args.checkpoint.resolve()), 'export_sha256': identity.get('export_sha256'),
              'seed': args.seed, 'denoising_steps': args.denoising_steps, 'ahead_s': AHEAD_S, 'inference_rate_hz': rate, 'samples': []}
    pairs = []
    with torch.no_grad():
        for k, (i, path) in enumerate(zip(ids, inputs)):
            with np.load(path) as archive:
                image = np.asarray(archive['observation/image'])
                state = np.asarray(archive['observation/state'], np.float32)
            observation = make_joint_observation(image, state)
            out = get_action(cfg, model, stats, observation, PROMPT, seed=args.seed, randomize_seed=False,
                             num_denoising_steps_action=args.denoising_steps,
                             generate_future_state_and_value_in_parallel=True, batch_size=1)
            actions = threshold_jaw(out['actions'])
            dream = np.asarray(out['future_image_predictions']['future_image'], np.uint8)
            value = float(out['value_prediction'])
            later_id = None
            if np.isfinite(stamps[k]):
                ahead = np.flatnonzero(np.isfinite(stamps) & (stamps >= stamps[k] + AHEAD_S))
                if len(ahead):
                    later_id = ids[int(ahead[0])]
            later = frames.get(later_id)
            diff = np.abs(dream.astype(np.int16) - later.astype(np.int16)).astype(np.uint8) if later is not None else np.full_like(dream, 96)
            t = (stamps[k] - stamps[0]) if np.isfinite(stamps[k]) else float('nan')
            strip = label_strip(np.concatenate([image, dream, later if later is not None else np.full_like(dream, 96), diff], axis=1),
                                [f'live t={t:.1f}s (req {i})', f'dream +{AHEAD_S:.1f}s  v={value:.2f}',
                                 f'live +{AHEAD_S:.1f}s (req {later_id})' if later is not None else 'n/a', '|diff|'])
            Image.fromarray(strip).save(strips_dir / f'{i:06d}.png')
            pair = np.concatenate([image, dream], axis=1)
            panel = Image.fromarray(pair).resize((696, 348), Image.NEAREST)
            canvas = Image.new('RGB', (696, 366), (20, 20, 20))
            canvas.paste(panel, (0, 18))
            ImageDraw.Draw(canvas).text((4, 3), f'live  t={t:5.1f} s   req {i}', fill=(230, 230, 230))
            ImageDraw.Draw(canvas).text((352, 3), f'dream +{AHEAD_S:.1f} s   value {value:.2f}', fill=(230, 230, 230))
            encoder.stdin.write(np.asarray(canvas).tobytes())
            if k % args.contact_every == 0:
                pairs.append(label_strip(pair, [f't={t:.0f}s req {i}', f'dream v={value:.2f}']))
            record = {'request_id': i, 't_s': None if np.isnan(t) else float(t), 'value': value,
                      'offline_vs_live_first_row_mm': (float(np.linalg.norm(actions[0, :3] - live[i][0, :3]) * 1000) if i in live else None),
                      'later_request_id': later_id, 'dream_vs_later_l1': l1(dream, later) if later is not None else None,
                      'dream_vs_later_psnr': psnr(dream, later) if later is not None else None,
                      'copy_now_vs_later_l1': l1(image, later) if later is not None else None}
            report['samples'].append(record)
            if k % 25 == 0:
                print(f'[{k}/{len(ids)}] req {i} t={t:.1f}s value {value:.2f} '
                      f"first row {record['offline_vs_live_first_row_mm']} mm from live", flush=True)
    encoder.stdin.close()
    encoder.wait()
    columns = 3
    if pairs:
        w, h = pairs[0].shape[1], pairs[0].shape[0]
        rows = (len(pairs) + columns - 1) // columns
        sheet = Image.new('RGB', (columns * (w + 8) - 8, rows * (h + 8) - 8), (255, 255, 255))
        for n, pair in enumerate(pairs):
            sheet.paste(Image.fromarray(pair), ((n % columns) * (w + 8), (n // columns) * (h + 8)))
        sheet.save(args.output / 'dreams_contact.png')
    finite = lambda key: [s[key] for s in report['samples'] if s[key] is not None]  # noqa: E731
    report['summary'] = {key: (float(np.mean(finite(key))) if finite(key) else None) for key in
                         ('offline_vs_live_first_row_mm', 'dream_vs_later_l1', 'copy_now_vs_later_l1', 'value')}
    (args.output / 'dream_report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report['summary'], indent=2))
    print(f'Wrote {args.output}', flush=True)


if __name__ == '__main__':
    main()
