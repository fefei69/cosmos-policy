"""Local "dream" check for a Hanoi waypoint_v4 export, using a single-episode HDF5 extract.

The cluster evaluators require the full raw recording and an H100/H200. This script
runs the same inference path (``waypoint_policy.load_policy`` + ``cosmos_utils.get_action``)
on any CUDA GPU, reading observations from ``data/hanoi_cosmos/exports_local/hanoi_episode_040.h5``
and the ground-truth sparse targets from the prepared split archive. For each sample it
saves:

* ``NN_rowR_strip.png``   [current | predicted future | recorded future | |difference|]
* ``NN_rowR_dream.gif``   the full 25-frame video the model generated, decoded by the VAE
                          (frames 5-8 current RGB, 17-20 predicted future RGB, rest blank slots)
* ``dream_summary.gif``   all strips, one per frame
* ``report.json``         first-target error, jaw intent, value, future-state error, image metrics

This is a qualitative check of the world-model head plus action decode. It says nothing
about task success on hardware.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

DEFAULT_RUN = Path('data/hanoi_cosmos/runs/cosmos_policy/hanoi/hanoi_cosmos_waypoint_v4_20260917')
GAMMA = 0.9995
NONIMAGE_SLOTS = [0, 1, 3, 4, 6]  # blank, proprio, actions, future proprio, value


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return float('inf') if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def l1(a, b):
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def patch_rope(mode):
    """Force or auto-detect the unfused PyTorch RoPE when TE's fused kernel lacks this GPU's arch."""
    import cosmos_policy._src.predict2.networks.minimal_v4_dit as dit

    original = dit.apply_rotary_pos_emb
    state = {'mode': mode, 'fused': mode != 'unfused', 'fell_back': False}

    def wrapped(t, freqs, *args, **kwargs):
        if state['fused']:
            try:
                return original(t, freqs, *args, **kwargs)
            except Exception as error:  # noqa: BLE001 - any kernel/arch failure triggers the fallback
                if mode != 'auto':
                    raise
                state['fused'], state['fell_back'] = False, True
                print(f'Fused TE RoPE failed on this GPU ({type(error).__name__}: {error}); '
                      'switching to the unfused PyTorch path', flush=True)
        kwargs['fused'] = False
        return original(t, freqs, *args, **kwargs)

    dit.apply_rotary_pos_emb = wrapped
    return state


def label_strip(strip, labels, font_px=14):
    """Add a text bar above each 224-wide panel."""
    from PIL import Image, ImageDraw

    height, width = strip.shape[:2]
    canvas = Image.new('RGB', (width, height + font_px + 6), (20, 20, 20))
    canvas.paste(Image.fromarray(strip), (0, font_px + 6))
    draw = ImageDraw.Draw(canvas)
    panel = width // len(labels)
    for i, text in enumerate(labels):
        draw.text((i * panel + 4, 2), text, fill=(230, 230, 230))
    return np.asarray(canvas)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_RUN / 'exports/iter_000002000.pt')
    parser.add_argument('--metadata', type=Path, default=Path('data/hanoi_cosmos/waypoint_v4'))
    parser.add_argument('--embeddings', default='data/hanoi_cosmos/t5_embeddings.pkl')
    parser.add_argument('--episode-h5', type=Path, default=Path('data/hanoi_cosmos/exports_local/hanoi_episode_040.h5'))
    parser.add_argument('--episode', type=int, default=40, help='Episode index the extract holds')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--samples', type=int, default=6, help='Samples spread evenly across the episode')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--denoising-steps', type=int, default=5)
    parser.add_argument('--rope', choices=['auto', 'fused', 'unfused'], default='auto')
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    output = args.output_dir or Path('data/hanoi_cosmos/evals/dream_local') / args.checkpoint.stem
    if output.exists() and not args.overwrite:
        raise FileExistsError(f'Refusing to overwrite {output}; pass --overwrite or --output-dir')

    # Platform constants bind at import time; the waypoint model uses the joint dimensions (7 / 8 / 4).
    os.environ['COSMOS_POLICY_PLATFORM'] = 'hanoi_joint'
    os.environ.setdefault('IMAGINAIRE_OUTPUT_ROOT', str(Path('data/hanoi_cosmos/runs').resolve()))
    import h5py
    import imageio.v2 as imageio
    import torch
    from PIL import Image

    from cosmos_policy.datasets.hanoi_joint_data import PROMPT, read_archive
    from cosmos_policy.datasets.hanoi_waypoint_data import CONTRACT
    from cosmos_policy.experiments.robot.cosmos_utils import (
        extract_action_chunk_from_latent_sequence, get_action, undo_latent_injection,
    )
    from cosmos_policy.experiments.robot.hanoi.joint_policy import absolute_joint_actions, make_joint_observation
    from cosmos_policy.experiments.robot.hanoi.waypoint_policy import inference_config_for, load_policy

    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required')
    gpu = torch.cuda.get_device_name()
    capability = torch.cuda.get_device_capability()

    # ---- samples: prepared split archive -> rows inside the single-episode extract ----
    contract = json.loads((args.metadata / 'metadata.json').read_text())['contract']
    if contract != CONTRACT:
        raise ValueError(f'Expected {CONTRACT}, got {contract}')
    archive = read_archive(args.metadata / f'{args.split}.npz')
    in_episode = np.flatnonzero(archive['episode_indices'] == args.episode)
    if not len(in_episode):
        raise ValueError(f'Episode {args.episode} has no samples in the {args.split} split')
    in_episode = in_episode[np.argsort(archive['source_observation_indices'][in_episode])]
    lo, hi = (int(v) for v in archive['source_episode_bounds'][in_episode[0]])
    picks = in_episode[np.unique(np.linspace(0, len(in_episode) - 1, args.samples).round().astype(int))]

    # Archive rows are absolute indices into the raw recording. ``shift`` maps them onto
    # the file at hand: a single-episode extract starts at 0, the raw recording does not.
    h5 = h5py.File(args.episode_h5, 'r')
    offsets, lengths = np.asarray(h5['ep_offset']), np.asarray(h5['ep_len'])
    if len(offsets) == 1:
        if int(lengths[0]) != hi - lo or int(offsets[0]) != 0:
            raise ValueError('Episode extract does not match the split archive episode bounds')
        shift = lo
    else:
        if int(offsets[args.episode]) != lo or int(lengths[args.episode]) != hi - lo:
            raise ValueError('Raw recording episode bounds do not match the split archive')
        shift = 0

    # ---- model ----
    rope = patch_rope(args.rope)
    cfg = inference_config_for(contract, args.checkpoint, args.metadata / 'dataset_statistics.json', args.embeddings)
    cfg.num_denoising_steps_action = args.denoising_steps
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, stats, _ = load_policy(cfg, contract)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    print(f'Loaded policy on {gpu} (sm_{capability[0]}{capability[1]}) in {load_seconds:.1f}s; '
          f'peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB', flush=True)

    output.mkdir(parents=True, exist_ok=True)
    report = {'checkpoint': str(args.checkpoint.resolve()), 'contract': contract, 'split': args.split,
              'episode': args.episode, 'episode_h5': str(args.episode_h5.resolve()), 'gpu': gpu,
              'compute_capability': f'{capability[0]}.{capability[1]}', 'seed': args.seed,
              'denoising_steps': args.denoising_steps, 'load_seconds': load_seconds, 'samples': []}
    strips = []
    with torch.no_grad():
        for n, k in enumerate(picks):
            row = int(archive['source_observation_indices'][k]) - shift
            future_row = min(int(archive['source_action_indices'][k][-1]) + 1, hi - 1) - shift
            image = np.asarray(h5['pixels'][row])
            state = np.r_[h5['joint_positions'][row], h5['proprio'][row, 6]].astype(np.float32)
            xyz = np.asarray(h5['proprio'][row, :3], np.float32)
            if not (np.allclose(state, archive['states'][k]) and np.allclose(xyz, archive['cartesian_positions'][k])):
                raise ValueError(f'Extract row {row} disagrees with the archive sample {k}')
            recorded_future = np.asarray(h5['pixels'][future_row])
            recorded_future_state = np.r_[h5['joint_positions'][future_row], h5['proprio'][future_row, 6]].astype(np.float32)
            target = archive['actions'][k]
            pads = archive['actions_is_pad'][k]

            observation = make_joint_observation(image, state)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = get_action(cfg, model, stats, observation, PROMPT, seed=args.seed, randomize_seed=False,
                             num_denoising_steps_action=args.denoising_steps,
                             generate_future_state_and_value_in_parallel=True, batch_size=1)
            torch.cuda.synchronize()
            latency = time.perf_counter() - t0

            actions = absolute_joint_actions(out['actions'], xyz)
            predicted_future = np.asarray(out['future_image_predictions']['future_image'])
            latent = out['generated_latent']
            # Predicted future proprio lives in latent slot 4, injected the same way as actions.
            slot = torch.full((1,), 4, dtype=torch.int64, device=latent.device)
            future_state_norm = extract_action_chunk_from_latent_sequence(latent, (1, 7), slot).float().cpu().numpy()[0, 0]
            predicted_future_state = 0.5 * (future_state_norm + 1) * (stats['proprio_max'] - stats['proprio_min']) + stats['proprio_min']
            # Full 25-frame decode of what the model generated, with non-image slots reset to their blank placeholders.
            clean = undo_latent_injection(latent.clone(), out['orig_clean_latent_frames'], NONIMAGE_SLOTS)
            dream = ((model.decode(clean) + 1) * 127.5).clamp(0, 255)[0].permute(1, 2, 3, 0).to(torch.uint8).cpu().numpy()

            errors_mm = np.linalg.norm(actions[:, :3] - target[:, :3], axis=-1) * 1000
            valid = ~pads
            diff = np.abs(predicted_future.astype(np.int16) - recorded_future.astype(np.int16)).astype(np.uint8)
            strip = label_strip(np.concatenate([image, predicted_future, recorded_future, diff], axis=1),
                                [f'current (row {row})', 'predicted future', f'recorded future (row {future_row})', '|diff|'])
            name = f'{n:02d}_row{row}'
            Image.fromarray(strip).save(output / f'{name}_strip.png')
            imageio.mimsave(output / f'{name}_dream.gif', list(dream), duration=0.25, loop=0)
            strips.append(strip)

            record = {
                'index': n, 'archive_index': int(k), 'observation_row': row, 'future_row': future_row,
                'rows_ahead': future_row - row, 'latency_seconds': latency,
                'first_target_xyz_mm': float(errors_mm[0]),
                'first_jaw_predicted': int(actions[0, 3]), 'first_jaw_target': int(target[0, 3]),
                'first_hit_5mm': bool(errors_mm[0] <= 5 and actions[0, 3] == target[0, 3]),
                'valid_horizon_xyz_mm': float(errors_mm[valid].mean()),
                'jaw_accuracy_valid': float(np.mean(actions[valid, 3] == target[valid, 3])),
                'value_predicted': float(out['value_prediction']),
                'value_true': float(GAMMA ** (hi - 1 - (future_row + shift))),
                'future_joint_error_rad': float(np.linalg.norm(predicted_future_state[:6] - recorded_future_state[:6])),
                'future_jaw_error_m': float(abs(predicted_future_state[6] - recorded_future_state[6])),
                'predicted_vs_recorded_l1': l1(predicted_future, recorded_future),
                'predicted_vs_recorded_psnr': psnr(predicted_future, recorded_future),
                'copy_current_l1': l1(image, recorded_future),
                'copy_current_psnr': psnr(image, recorded_future),
                'predicted_targets_abs': actions.round(4).tolist(),
                'recorded_targets_abs': target.round(4).tolist(),
                'strip_png': f'{name}_strip.png', 'dream_gif': f'{name}_dream.gif',
            }
            report['samples'].append(record)
            print(f"[{n}] row {row:>5} -> {future_row:>5} | first target {errors_mm[0]:6.1f} mm, jaw "
                  f"{int(actions[0, 3])}/{int(target[0, 3])} | value {record['value_predicted']:.3f} vs "
                  f"{record['value_true']:.3f} | future img L1 {record['predicted_vs_recorded_l1']:.1f} "
                  f"(copy-current {record['copy_current_l1']:.1f}) | {latency:.2f}s", flush=True)
    h5.close()

    imageio.mimsave(output / 'dream_summary.gif', strips, duration=1.5, loop=0)
    keys = ('first_target_xyz_mm', 'valid_horizon_xyz_mm', 'jaw_accuracy_valid', 'latency_seconds', 'rows_ahead',
            'future_joint_error_rad', 'future_jaw_error_m', 'predicted_vs_recorded_l1', 'copy_current_l1')
    finite = lambda key: [s[key] for s in report['samples'] if np.isfinite(s[key])]  # noqa: E731
    report['summary'] = {key: float(np.mean(finite(key))) for key in keys}
    report['summary']['first_hit_rate_5mm'] = float(np.mean([s['first_hit_5mm'] for s in report['samples']]))
    report['summary']['value_abs_error'] = float(np.mean([abs(s['value_predicted'] - s['value_true']) for s in report['samples']]))
    report['summary']['fraction_better_than_copy_l1'] = float(np.mean(
        [s['predicted_vs_recorded_l1'] < s['copy_current_l1'] for s in report['samples']]))
    report['rope'] = {'mode': rope['mode'], 'fused_used': rope['fused'] and not rope['fell_back'], 'fell_back': rope['fell_back']}
    report['peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({**report['summary'], 'rope': report['rope'], 'peak_allocated_gib': report['peak_allocated_gib']}, indent=2))
    print(f'Wrote {output}', flush=True)


if __name__ == '__main__':
    main()
