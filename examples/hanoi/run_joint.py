"""Bounded, resumable Cosmos training for the audited joint/sparse contract."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from cosmos_policy.datasets.hanoi_joint_data import CONTRACT, sha256
from cosmos_policy.utils.hanoi_checkpoint import checkpoint_iteration, latest_complete_checkpoint
from examples.hanoi.run_long import scratch_headroom
from examples.hanoi.run_pilot import execute_before


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualify-only', action='store_true', help='Two updates, reload/resume one update, export and inference; then stop')
    parser.add_argument('--run-name', default=os.environ.get('HANOI_JOINT_RUN_NAME', 'hanoi_cosmos_joint_sparse_20260917'))
    args = parser.parse_args()
    if not args.run_name.startswith('hanoi_cosmos_joint_') or Path(args.run_name).name != args.run_name:
        raise ValueError('Use a distinct hanoi_cosmos_joint_* run name')
    root = Path(__file__).resolve().parents[2]
    metadata = Path(os.environ.get('HANOI_JOINT_METADATA', str(root / 'data/hanoi_cosmos/joint_sparse_v3'))).resolve()
    prepared = json.loads((metadata / 'metadata.json').read_text())
    if prepared['contract'] != CONTRACT:
        raise ValueError('Wrong prepared task')
    import torch
    if torch.cuda.device_count() != 1 or 'H100' not in torch.cuda.get_device_name():
        raise RuntimeError('This experiment is fixed to one H100')
    run = root / 'data/hanoi_cosmos/runs/cosmos_policy/hanoi' / args.run_name
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ.update(COSMOS_POLICY_PLATFORM='hanoi_joint', HANOI_JOINT_RUN_NAME=args.run_name,
                      HANOI_JOINT_METADATA=str(metadata), HANOI_TRAINING_SCHEDULE='aloha')
    os.environ.pop('HANOI_CONTINUATION_ANCHOR', None)
    # Public Cosmos initialization is deliberate; the previous dense Cartesian
    # Hanoi checkpoint is incompatible with this observation/target contract.
    os.environ['HANOI_INIT_CHECKPOINT'] = str(root / 'checkpoints/public/Cosmos-Policy-LIBERO-Predict2-2B.pt')
    code_paths = [root / path for path in (
        'cosmos_policy/constants.py', 'cosmos_policy/config/hanoi_joint_config.py', 'cosmos_policy/config/hanoi_config.py',
        'cosmos_policy/datasets/hanoi_joint_data.py', 'cosmos_policy/datasets/hanoi_joint_dataset.py',
        'cosmos_policy/models/hanoi_model.py', 'cosmos_policy/models/policy_text2world_model.py',
        'cosmos_policy/models/policy_video2world_model.py', 'cosmos_policy/utils/hanoi_joint_training.py',
        'cosmos_policy/utils/hanoi_optimizer.py', 'cosmos_policy/utils/hanoi_training.py',
        'cosmos_policy/utils/hanoi_schedule.py', 'cosmos_policy/datasets/resumable_sampler.py',
        'cosmos_policy/trainer.py', 'cosmos_policy/scripts/train.py', 'cosmos_policy/datasets/hanoi_data.py',
        'cosmos_policy/experiments/robot/cosmos_utils.py', 'cosmos_policy/experiments/robot/hanoi/policy.py',
        'cosmos_policy/experiments/robot/hanoi/joint_policy.py',
        'cosmos_policy/experiments/robot/hanoi/run_hanoi_joint_eval.py')]
    identity = {'contract': CONTRACT, 'metadata_sha256': sha256(metadata / 'metadata.json'),
                'statistics_sha256': prepared['statistics_sha256'], 'raw_sha256': prepared['raw_sha256'],
                'initial_weights': os.environ['HANOI_INIT_CHECKPOINT'], 'effective_batch_size': 32,
                'gpus': 1, 'gpu_model': 'H100', 'max_updates': 30000, 'save_every': 2000,
                'code_sha256': {str(p.relative_to(root)): sha256(p) for p in code_paths}}
    contract_path = run / 'joint_contract.json'
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != identity:
            raise ValueError('Refusing resume with changed joint data, model code or training settings')
    else:
        if (run / 'checkpoints/latest_checkpoint.txt').exists():
            raise ValueError('Existing checkpoints lack the joint contract identity')
        atomic_json(contract_path, identity)
    allocation = int(os.environ.get('HANOI_JOINT_ALLOCATION_SECONDS', '172800'))
    deadline = time.time() + allocation - 60
    stop_at = deadline - (240 if args.qualify_only else 2400)
    os.environ['HANOI_STOP_AT_EPOCH'] = str(stop_at)
    python = str(root / '.venv/bin/python')
    train = [python, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=1',
             '-m', 'cosmos_policy.scripts.train', '--config=cosmos_policy/config/hanoi_joint_config.py',
             '--', 'experiment=cosmos_predict2_2b_hanoi_joint']
    journal = {'run': str(run), 'job_id': os.environ.get('SLURM_JOB_ID'), 'events': [], 'qualification_only': args.qualify_only}
    def record(phase, **fields):
        journal['phase'] = phase
        journal['events'].append({'time': time.time(), 'phase': phase, **fields})
        atomic_json(run / 'joint_pipeline.json', journal)
        print('HANOI_JOINT ' + json.dumps(journal['events'][-1]), flush=True)
    def execute(phase, command):
        record(phase, command=command)
        execute_before(command, root, deadline)
    def export(checkpoint):
        target = run / 'exports' / (checkpoint.name + '.pt')
        if not target.exists():
            execute('export', [python, 'examples/hanoi/export_checkpoint.py', '--checkpoint', str(checkpoint), '--output', str(target)])
        return target
    def evaluate(checkpoint, output, *, mode='loss', samples=0, split='val', selection=None):
        if output.exists():
            value = json.loads(output.read_text())
            if value['checkpoint'] != str(checkpoint.resolve()) or value['split'] != split or value['mode'] != mode:
                raise ValueError('Existing evaluation identity differs')
            return value
        command = [python, '-m', 'cosmos_policy.experiments.robot.hanoi.run_hanoi_joint_eval',
                   '--checkpoint', str(checkpoint), '--metadata', str(metadata), '--mode', mode,
                   '--samples', str(samples), '--split', split, '--output', str(output)]
        if selection:
            command += ['--selection', str(selection)]
        execute(f'evaluate_{split}', command)
        return json.loads(output.read_text())
    def prune_older(checkpoint):
        # Only this new run's generated older full checkpoints, after their
        # validated inference export and the newest complete checkpoint exist.
        for old in sorted((run / 'checkpoints').glob('iter_*')):
            if old.resolve().parent != (run / 'checkpoints').resolve() or old.is_symlink():
                raise ValueError('Checkpoint path escapes run')
            if checkpoint_iteration(old) >= checkpoint_iteration(checkpoint):
                continue
            if (run / 'exports' / (old.name + '.pt')).exists() or checkpoint_iteration(old) in (2, 3):
                record('prune_old_resumable', path=str(old), retained=str(checkpoint))
                shutil.rmtree(old)
    try:
        quota = scratch_headroom()
        if quota['free_bytes_lower_bound'] < 160_000_000_000:
            raise RuntimeError('Need 160 GB scratch headroom for retained exports and two in-flight full checkpoints')
        record('preflight_passed', quota=quota)
        checkpoint = latest_complete_checkpoint(run) if (run / 'checkpoints/latest_checkpoint.txt').exists() else None
        qualification = run / 'qualification.json'
        if not qualification.exists():
            if checkpoint is None:
                execute('qualify_train', train + ['trainer.max_iter=2', 'checkpoint.save_iter=2', 'trainer.max_val_iter=2'])
                checkpoint = latest_complete_checkpoint(run)
            if checkpoint_iteration(checkpoint) == 2:
                execute('qualify_resume', train + ['trainer.max_iter=3', 'checkpoint.save_iter=3', 'trainer.max_val_iter=2'])
                checkpoint = latest_complete_checkpoint(run)
            if checkpoint_iteration(checkpoint) != 3:
                raise RuntimeError('Qualification requires exactly three completed updates after the step-two resume')
            exported = export(checkpoint)
            result = evaluate(exported, run / 'qualification_validation.json', mode='both', samples=2)
            if not result.get('serving_parity_passed'):
                raise RuntimeError('Serving parity qualification failed')
            atomic_json(qualification, {'passed': True, 'checkpoint': str(checkpoint), 'resume_step': checkpoint_iteration(checkpoint),
                                       'report': str(run / 'qualification_validation.json')})
            prune_older(checkpoint)
        if args.qualify_only:
            record('qualification_complete', checkpoint=str(checkpoint))
            return
        exports = []
        for step in range(2000, 30001, 2000):
            if checkpoint_iteration(checkpoint) < step:
                if time.time() >= stop_at:
                    record('stopped_for_allocation_budget', checkpoint=str(checkpoint))
                    return
                if scratch_headroom()['free_bytes_lower_bound'] < 100_000_000_000:
                    raise RuntimeError('Insufficient storage headroom before next training stage')
                execute('train', train + [f'trainer.max_iter={step}'])
                checkpoint = latest_complete_checkpoint(run)
                if checkpoint_iteration(checkpoint) < step:
                    record('stopped_for_allocation_budget', checkpoint=str(checkpoint))
                    return
                exported = export(checkpoint)
                prune_older(checkpoint)
            else:
                exported = run / 'exports' / f'iter_{step:09d}.pt'
                if not exported.exists():
                    old = run / 'checkpoints' / f'iter_{step:09d}'
                    exported = export(old)
            exports.append(exported)
        reports = [(evaluate(path, run / 'evaluation' / f'{path.stem}_validation_loss.json'), path) for path in exports]
        best, best_path = min(reports, key=lambda item: (item[0]['denoising_metrics']['loss'], item[1].name))
        selected = {'checkpoint': str(best_path.resolve()), 'rule': 'minimum full-validation fixed-noise joint denoising loss; earlier step breaks ties',
                    'validation_loss': best['denoising_metrics']['loss'], 'test_consulted_for_selection': False}
        selection_path = run / 'selection.json'
        if selection_path.exists() and json.loads(selection_path.read_text()) != selected:
            raise ValueError('A different checkpoint selection is already locked')
        atomic_json(selection_path, selected)
        evaluate(best_path, run / 'selected_validation.json', mode='both')
        evaluate(best_path, run / 'selected_test.json', mode='both', split='test', selection=selection_path)
        record('completed', **selected)
    except BaseException as error:
        record('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
