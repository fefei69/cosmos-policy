"""Regenerate the model's "dreams" for a recorded deployment run, offline.

The live server returns only the eight destinations unless started with ``--dream``, but the
client saves every request's observation under ``<run>/inference_inputs/NNNNNN.npz``. This script
re-runs the policy on those observations with the future-frame decode enabled and writes, under
``<run>/dreams/`` (or ``--output-dir``):

* ``NNNNNN_strip.png``   [live now | dreamed future | live after 8 waypoints | |difference|]
* ``NNNNNN_dream.gif``   with ``--gifs``: the full 25-frame decode (frames 5-8 current, 17-20 future)
* ``dreams_contact.png`` every [now | dream] pair on one sheet, with the chosen destination
* ``report.json``        per request: offline vs live first destination (mm), value, image metrics

The dream is the frame the model expects once the whole eight-waypoint chunk has been executed
(about 21 s ahead in the recording). The closest live comparator is the observation eight requests
later, shown in the third panel when the run got that far. Offline and live decisions agree to
about 0.2 mm (bf16 across processes), so these are the dreams behind each live destination.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from dream_local import DEFAULT_RUN, NONIMAGE_SLOTS, l1, label_strip, patch_rope, psnr

CHUNK = 8


def phase(z_m):
    return 'grasp' if z_m < 0.10 else ('release' if z_m < 0.17 else 'hover')


def peg(y_m):
    return 'A' if y_m < -0.02 else ('B' if y_m < 0.05 else 'C')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dir', type=Path, required=True, help='OpenPI client run directory')
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_RUN / 'exports/iter_000008000.pt')
    parser.add_argument('--metadata', type=Path, default=Path('data/hanoi_cosmos/waypoint_v4'))
    parser.add_argument('--embeddings', default='data/hanoi_cosmos/t5_embeddings.pkl')
    parser.add_argument('--requests', type=int, nargs='*', default=None, help='Request ids to dream (default all)')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--denoising-steps', type=int, default=5)
    parser.add_argument('--rope', choices=['auto', 'fused', 'unfused'], default='auto')
    parser.add_argument('--gifs', action='store_true', help='Also decode the full 25-frame video per request')
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    run = args.run_dir
    inputs = sorted((run / 'inference_inputs').glob('*.npz'))
    if not inputs:
        raise FileNotFoundError(f'No inference_inputs/*.npz under {run}')
    if args.requests is not None:
        wanted = set(args.requests)
        inputs = [p for p in inputs if int(p.stem) in wanted]
    output = args.output_dir or run / 'dreams'
    if output.exists() and not args.overwrite:
        raise FileExistsError(f'Refusing to overwrite {output}; pass --overwrite or --output-dir')
    live = {}
    events = run / 'events.jsonl'
    if events.is_file():
        for line in events.read_text().splitlines():
            event = json.loads(line)
            if event.get('event') == 'prediction':
                live[int(event['request_id'])] = np.asarray(event['actions'], np.float64)
    server_metadata = run / 'server_metadata.json'
    if server_metadata.is_file():
        identity = json.loads(server_metadata.read_text())['cosmos_hanoi']
        if identity.get('seed', args.seed) != args.seed or identity.get('num_denoising_steps', args.denoising_steps) != args.denoising_steps:
            print(f"Warning: the run used seed {identity.get('seed')} and {identity.get('num_denoising_steps')} steps", flush=True)

    os.environ['COSMOS_POLICY_PLATFORM'] = 'hanoi_joint'
    os.environ.setdefault('IMAGINAIRE_OUTPUT_ROOT', str(Path('data/hanoi_cosmos/runs').resolve()))
    import imageio.v2 as imageio
    import torch
    from PIL import Image

    from cosmos_policy.datasets.hanoi_joint_data import PROMPT
    from cosmos_policy.datasets.hanoi_waypoint_data import CONTRACT
    from cosmos_policy.experiments.robot.cosmos_utils import get_action, undo_latent_injection
    from cosmos_policy.experiments.robot.hanoi.joint_policy import absolute_joint_actions, make_joint_observation
    from cosmos_policy.experiments.robot.hanoi.waypoint_policy import inference_config_for, load_policy

    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required')
    contract = json.loads((args.metadata / 'metadata.json').read_text())['contract']
    if contract != CONTRACT:
        raise ValueError(f'Expected {CONTRACT}, got {contract}')
    rope = patch_rope(args.rope)
    cfg = inference_config_for(contract, args.checkpoint, args.metadata / 'dataset_statistics.json', args.embeddings)
    cfg.num_denoising_steps_action = args.denoising_steps
    started = time.perf_counter()
    model, stats, _ = load_policy(cfg, contract)
    torch.cuda.synchronize()
    print(f'Loaded policy on {torch.cuda.get_device_name()} in {time.perf_counter() - started:.1f}s', flush=True)

    output.mkdir(parents=True, exist_ok=True)
    frames = {int(p.stem): np.load(p)['observation/image'] for p in sorted((run / 'inference_inputs').glob('*.npz'))}
    report = {'run_dir': str(run.resolve()), 'checkpoint': str(args.checkpoint.resolve()), 'seed': args.seed,
              'denoising_steps': args.denoising_steps, 'samples': []}
    pairs, strips = [], []
    blank = np.full((224, 224, 3), 96, np.uint8)
    with torch.no_grad():
        for path in inputs:
            request = int(path.stem)
            with np.load(path) as archive:
                image = np.asarray(archive['observation/image'])
                state = np.asarray(archive['observation/state'], np.float32)
                xyz = np.asarray(archive['observation/cartesian_position'], np.float32)
            observation = make_joint_observation(image, state)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = get_action(cfg, model, stats, observation, PROMPT, seed=args.seed, randomize_seed=False,
                             num_denoising_steps_action=args.denoising_steps,
                             generate_future_state_and_value_in_parallel=True, batch_size=1)
            torch.cuda.synchronize()
            latency = time.perf_counter() - t0
            actions = absolute_joint_actions(out['actions'], xyz)
            dream = np.asarray(out['future_image_predictions']['future_image'], np.uint8)
            value = float(out['value_prediction'])
            later = frames.get(request + CHUNK)
            chosen = actions[0]
            title = f'req {request}: {phase(chosen[2])} {peg(chosen[1])} ({chosen[0]*1000:.0f},{chosen[1]*1000:.0f},{chosen[2]*1000:.0f}) v={value:.2f}'
            diff = (np.abs(dream.astype(np.int16) - later.astype(np.int16)).astype(np.uint8) if later is not None else blank)
            strip = label_strip(np.concatenate([image, dream, later if later is not None else blank, diff], axis=1),
                                [f'live now (req {request})', 'dreamed future', f'live after {CHUNK} waypoints' if later is not None else 'n/a', '|diff|'])
            Image.fromarray(strip).save(output / f'{request:06d}_strip.png')
            strips.append(strip)
            pairs.append(label_strip(np.concatenate([image, dream], axis=1), [title, '']))
            if args.gifs:
                clean = undo_latent_injection(out['generated_latent'].clone(), out['orig_clean_latent_frames'], NONIMAGE_SLOTS)
                video = ((model.decode(clean) + 1) * 127.5).clamp(0, 255)[0].permute(1, 2, 3, 0).to(torch.uint8).cpu().numpy()
                imageio.mimsave(output / f'{request:06d}_dream.gif', list(video), duration=0.25, loop=0)
            record = {
                'request_id': request, 'latency_seconds': latency, 'value_predicted': value,
                'first_destination_abs': chosen.round(4).tolist(), 'phase': phase(chosen[2]), 'peg': peg(chosen[1]),
                'predicted_targets_abs': actions.round(4).tolist(),
                'offline_vs_live_first_mm': (float(np.linalg.norm(chosen[:3] - live[request][0, :3]) * 1000) if request in live else None),
                'dream_vs_later_l1': l1(dream, later) if later is not None else None,
                'dream_vs_later_psnr': psnr(dream, later) if later is not None else None,
                'copy_now_vs_later_l1': l1(image, later) if later is not None else None,
                'strip_png': f'{request:06d}_strip.png',
            }
            report['samples'].append(record)
            same = f"{record['offline_vs_live_first_mm']:.2f} mm from live" if record['offline_vs_live_first_mm'] is not None else 'no live record'
            l1s = (f"dream L1 {record['dream_vs_later_l1']:.1f} vs copy-now {record['copy_now_vs_later_l1']:.1f}"
                   if later is not None else 'no later frame')
            print(f'[{request:3d}] {title:<44} {same:>18} | {l1s} | {latency:.2f}s', flush=True)

    columns = 3
    width, height = pairs[0].shape[1], pairs[0].shape[0]
    rows = (len(pairs) + columns - 1) // columns
    sheet = Image.new('RGB', (columns * (width + 8) - 8, rows * (height + 8) - 8), (255, 255, 255))
    for k, pair in enumerate(pairs):
        sheet.paste(Image.fromarray(pair), ((k % columns) * (width + 8), (k // columns) * (height + 8)))
    sheet.save(output / 'dreams_contact.png')
    imageio.mimsave(output / 'dream_summary.gif', strips, duration=1.0, loop=0)
    finite = lambda key: [s[key] for s in report['samples'] if s[key] is not None]  # noqa: E731
    report['summary'] = {key: (float(np.mean(finite(key))) if finite(key) else None) for key in
                         ('latency_seconds', 'offline_vs_live_first_mm', 'dream_vs_later_l1', 'copy_now_vs_later_l1')}
    if finite('offline_vs_live_first_mm'):
        report['summary']['offline_vs_live_first_mm_max'] = float(np.max(finite('offline_vs_live_first_mm')))
    report['rope'] = {'mode': rope['mode'], 'fused_used': rope['fused'] and not rope['fell_back'], 'fell_back': rope['fell_back']}
    report['peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({**report['summary'], 'peak_allocated_gib': report['peak_allocated_gib']}, indent=2))
    print(f'Wrote {output}', flush=True)


if __name__ == '__main__':
    main()
