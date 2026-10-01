"""Bounded, resumable Cosmos training on the six-task dense labels (contract hanoi_multitask_v6).

Stage structure as the dense launcher: qualify (two updates, save, reload,
one update, export, inference with every task prompt in the cache), then
stages of 2,000 updates to 32,000, each followed by an export and the section
7 evaluation on validation rows with each row's own prompt. The final
checkpoint is chosen by decision 11 over all tasks (lowest mean per-step XYZ
error with jaw accuracy at least 0.99, ties by earlier step), then evaluated
once on the test split; both final passes include the language-following
probe. ``--init video`` starts from the Cosmos-Predict2 video base, ``--init
libero`` from the LIBERO policy checkpoint. Runs use distinct run names and
never share a directory.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import time

from cosmos_policy.config.hanoi_dense_config import microbatch_from_env
from cosmos_policy.config.hanoi_multitask_config import EXPERIMENT, MAX_UPDATES, SAVE_EVERY
from cosmos_policy.datasets.hanoi_dense_data import HORIZON
from cosmos_policy.datasets.hanoi_joint_data import sha256
from cosmos_policy.datasets.hanoi_multitask_data import CONTRACT, DEFAULT_EMBEDDINGS, DEFAULT_METADATA, TASKS
from cosmos_policy.utils.hanoi_checkpoint import checkpoint_iteration, latest_complete_checkpoint
from examples.hanoi.run_dense import JAW_ACCURACY_FLOOR, QUALIFICATION_PARITY_TOLERANCE_MM, SELECTION_RULE, select_checkpoint
from examples.hanoi.run_joint import atomic_json
from examples.hanoi.run_long import scratch_headroom
from examples.hanoi.run_pilot import execute_before

ACCEPTED_GPUS = ('H100', 'H200')
DATE = '20260926'
INITS = {
    'libero': {'format': 'policy', 'default_path': 'checkpoints/public/Cosmos-Policy-LIBERO-Predict2-2B.pt', 'run': 'L'},
    'video': {'format': 'video_base', 'default_path': 'checkpoints/public/model-480p-16fps.pt', 'run': 'V'},
}
STAGE_EVAL = {'stride': 9, 'steps': 5}          # every export: about 5,100 validation rows across the six tasks
FINAL_EVAL = {'stride': 3, 'steps': 5, 'also_steps': 10, 'probe_rows': 64}  # selected export: about 15,000 rows, both step counts, language probe
CODE_PATHS = (
    'cosmos_policy/datasets/hanoi_multitask_data.py', 'cosmos_policy/datasets/hanoi_multitask_dataset.py',
    'cosmos_policy/config/hanoi_multitask_config.py', 'cosmos_policy/experiments/robot/hanoi/multitask_policy.py',
    'cosmos_policy/experiments/robot/hanoi/run_hanoi_multitask_eval.py', 'examples/hanoi/run_multitask.py',
    'cosmos_policy/constants.py', 'cosmos_policy/config/hanoi_config.py', 'cosmos_policy/config/hanoi_dense_config.py',
    'cosmos_policy/datasets/hanoi_data.py', 'cosmos_policy/datasets/hanoi_joint_data.py', 'cosmos_policy/datasets/hanoi_joint_dataset.py',
    'cosmos_policy/datasets/hanoi_dense_data.py', 'cosmos_policy/datasets/hanoi_dense_dataset.py',
    'cosmos_policy/models/hanoi_model.py', 'cosmos_policy/models/hanoi_dense_model.py',
    'cosmos_policy/models/policy_text2world_model.py', 'cosmos_policy/models/policy_video2world_model.py',
    'cosmos_policy/utils/hanoi_joint_training.py', 'cosmos_policy/utils/hanoi_waypoint_training.py',
    'cosmos_policy/utils/hanoi_optimizer.py', 'cosmos_policy/utils/hanoi_training.py', 'cosmos_policy/utils/hanoi_schedule.py',
    'cosmos_policy/datasets/resumable_sampler.py', 'cosmos_policy/trainer.py', 'cosmos_policy/scripts/train.py',
    'cosmos_policy/experiments/robot/cosmos_utils.py', 'cosmos_policy/experiments/robot/hanoi/policy.py',
    'cosmos_policy/experiments/robot/hanoi/joint_policy.py', 'cosmos_policy/experiments/robot/hanoi/dense_policy.py',
    'cosmos_policy/experiments/robot/hanoi/run_hanoi_dense_eval.py', 'examples/hanoi/run_dense.py',
)


def run_name_for(init):
    return f'hanoi_cosmos_multitask_{DATE}_{init}_init'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--init', choices=sorted(INITS), required=True)
    parser.add_argument('--qualify-only', action='store_true', help='Two updates, reload/resume one update, export and inference; then stop')
    parser.add_argument('--run-name', default=None)
    args = parser.parse_args()
    os.environ['HANOI_DENSE_HORIZON'] = str(HORIZON)  # chunk 16; before any import of cosmos_policy.constants; inherited by train/eval
    init = INITS[args.init]
    run_name = args.run_name or os.environ.get('HANOI_MULTITASK_RUN_NAME') or run_name_for(args.init)
    if not run_name.startswith('hanoi_cosmos_multitask_') or Path(run_name).name != run_name:
        raise ValueError('Use a distinct hanoi_cosmos_multitask_* run name')
    root = Path(__file__).resolve().parents[2]
    metadata = Path(os.environ.get('HANOI_MULTITASK_METADATA', str(root / DEFAULT_METADATA))).resolve()
    embeddings = Path(os.environ.get('HANOI_MULTITASK_EMBEDDINGS', str(root / DEFAULT_EMBEDDINGS))).resolve()
    prepared = json.loads((metadata / 'metadata.json').read_text())
    if prepared['contract'] != CONTRACT:
        raise ValueError('Wrong prepared task')
    if int(prepared['horizon']) != HORIZON:
        raise ValueError(f'Prepared dataset horizon {prepared["horizon"]} differs from the chunk of {HORIZON}')
    if [t['prompt'] for t in prepared['tasks']] != [t.prompt for t in TASKS]:
        raise ValueError('Prepared prompts differ from the task table in code')
    if not embeddings.is_file():
        raise FileNotFoundError(f'Prompt embedding cache is missing: {embeddings}')
    import pickle
    with embeddings.open('rb') as stream:
        cached = set(pickle.load(stream))
    absent = [t.direction for t in TASKS if t.prompt not in cached]
    if absent:
        raise ValueError(f'Embedding cache {embeddings} lacks the prompts of {absent}; encode with prepare_t5.py --prompt-set multitask')
    initial = Path(os.environ.get('HANOI_INIT_CHECKPOINT', str(root / init['default_path']))).resolve()
    if not initial.is_file():
        raise FileNotFoundError(f'Initial weights for run {init["run"]} ({args.init}) are missing: {initial}')
    import torch
    gpu = torch.cuda.get_device_name() if torch.cuda.device_count() else 'none'
    if torch.cuda.device_count() != 1 or not any(tag in gpu for tag in ACCEPTED_GPUS):
        raise RuntimeError(f'This experiment is fixed to one H100 or H200, not {gpu!r} x{torch.cuda.device_count()}')
    run = root / 'data/hanoi_cosmos/runs/cosmos_policy/hanoi' / run_name
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ.update(COSMOS_POLICY_PLATFORM='hanoi_dense', HANOI_MULTITASK_RUN_NAME=run_name, HANOI_MULTITASK_METADATA=str(metadata),
                      HANOI_MULTITASK_EMBEDDINGS=str(embeddings), HANOI_INIT_CHECKPOINT=str(initial), HANOI_INIT_FORMAT=init['format'])
    os.environ.pop('HANOI_CONTINUATION_ANCHOR', None)
    os.environ.pop('HANOI_TRAINING_SCHEDULE', None)
    os.environ.setdefault('HANOI_DENSE_MICROBATCH', '16')
    os.environ.setdefault('HANOI_DENSE_ACTIVATION_CHECKPOINT', 'none')
    print(f'Hashing initial weights {initial.name}...', flush=True)
    identity = {'contract': CONTRACT, 'horizon': HORIZON, 'metadata_sha256': sha256(metadata / 'metadata.json'),
                'prompts': [t.prompt for t in TASKS], 'embeddings_sha256': sha256(embeddings),
                'statistics_sha256': prepared['statistics_sha256'],
                'raw_sha256': {f['direction']: f['sha256'] for f in prepared['files']},  # one recording per task
                'initial_weights': str(initial), 'initial_weights_sha256': sha256(initial), 'initial_weights_format': init['format'],
                'run_label': init['run'], 'effective_batch_size': 32,
                'microbatch': microbatch_from_env(), 'activation_checkpoint': os.environ['HANOI_DENSE_ACTIVATION_CHECKPOINT'],
                'gpus': 1, 'gpu_class': 'H100 or H200', 'max_updates': MAX_UPDATES, 'save_every': SAVE_EVERY,
                'selection_rule': SELECTION_RULE, 'stage_evaluation': STAGE_EVAL, 'final_evaluation': FINAL_EVAL,
                'code_sha256': {path: sha256(root / path) for path in CODE_PATHS}}
    contract_path = run / 'joint_contract.json'
    if contract_path.exists():
        recorded = json.loads(contract_path.read_text())
        if recorded != identity:
            differing = {key for key in set(recorded) | set(identity) if recorded.get(key) != identity.get(key)}
            reason = os.environ.get('HANOI_DENSE_ACCEPT_CODE_CHANGES', '')
            if differing != {'code_sha256'} or not reason:
                raise ValueError('Refusing resume with changed dense data, model code, initial weights or training settings')
            # Only code hashes differ and the operator stated why: record each change beside the run, then continue.
            updates_path = run / 'code_updates.json'
            updates = json.loads(updates_path.read_text()) if updates_path.exists() else []
            for path in sorted(set(recorded['code_sha256']) | set(identity['code_sha256'])):
                if recorded['code_sha256'].get(path) != identity['code_sha256'].get(path):
                    updates.append({'time': time.time(), 'job_id': os.environ.get('SLURM_JOB_ID'), 'path': path,
                                    'old_sha256': recorded['code_sha256'].get(path), 'new_sha256': identity['code_sha256'].get(path),
                                    'reason': reason})
            atomic_json(updates_path, updates)
            atomic_json(contract_path, identity)
    else:
        if (run / 'checkpoints/latest_checkpoint.txt').exists():
            raise ValueError('Existing checkpoints lack the dense contract identity')
        atomic_json(contract_path, identity)
    if 'HANOI_DENSE_ALLOCATION_SECONDS' not in os.environ:
        raise RuntimeError('Export HANOI_DENSE_ALLOCATION_SECONDS from the batch script (the Slurm time limit)')
    allocation = int(os.environ['HANOI_DENSE_ALLOCATION_SECONDS'])
    deadline = time.time() + allocation - 60
    stop_at = deadline - (240 if args.qualify_only else 2700)  # room for one export and one stage evaluation
    os.environ['HANOI_STOP_AT_EPOCH'] = str(stop_at)
    python = str(root / '.venv/bin/python')
    train = [python, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=1',
             '-m', 'cosmos_policy.scripts.train', '--config=cosmos_policy/config/hanoi_multitask_config.py',
             '--', f'experiment={EXPERIMENT}']
    journal_path = run / 'dense_pipeline.json'
    journal = json.loads(journal_path.read_text()) if journal_path.exists() else {'run': str(run), 'events': []}
    journal.update({'job_id': os.environ.get('SLURM_JOB_ID'), 'gpu': gpu, 'init': args.init, 'qualification_only': args.qualify_only})
    def record(phase, **fields):
        journal['phase'] = phase
        journal['events'].append({'time': time.time(), 'phase': phase, 'job_id': os.environ.get('SLURM_JOB_ID'), **fields})
        atomic_json(journal_path, journal)
        print('HANOI_DENSE ' + json.dumps(journal['events'][-1]), flush=True)
    def execute(phase, command):
        record(phase, command=command)
        execute_before(command, root, deadline)
    def export(checkpoint):
        target = run / 'exports' / (checkpoint.name + '.pt')
        if not target.exists():
            execute('export', [python, 'examples/hanoi/export_checkpoint.py', '--checkpoint', str(checkpoint), '--output', str(target)])
        return target
    def evaluate(checkpoint, output, *, split='val', stride, steps, also_steps=0, future=False, parity=0, parity_tolerance=None,
                 probe_rows=0, selection=None):
        if output.exists():
            value = json.loads(output.read_text())
            if value['checkpoint'] != str(checkpoint.resolve()) or value['split'] != split or value['stride'] != stride:
                raise ValueError('Existing evaluation identity differs')
            return value
        command = [python, '-m', 'cosmos_policy.experiments.robot.hanoi.run_hanoi_multitask_eval',
                   '--checkpoint', str(checkpoint), '--metadata', str(metadata), '--embeddings', str(embeddings), '--split', split,
                   '--stride', str(stride), '--steps', str(steps), '--batch-size', '16', '--output', str(output)]
        if also_steps:
            command += ['--also-steps', str(also_steps)]
        if future:
            command += ['--future']
        if parity:
            command += ['--parity-samples', str(parity)]
        if parity_tolerance is not None:
            command += ['--parity-tolerance-mm', str(parity_tolerance)]
        if probe_rows:
            command += ['--probe-rows', str(probe_rows)]
        if selection:
            command += ['--selection', str(selection)]
        execute(f'evaluate_{split}', command)
        return json.loads(output.read_text())
    def prune_older(checkpoint):
        for old in sorted((run / 'checkpoints').glob('iter_*')):
            if old.resolve().parent != (run / 'checkpoints').resolve() or old.is_symlink():
                raise ValueError('Checkpoint path escapes run')
            if checkpoint_iteration(old) >= checkpoint_iteration(checkpoint):
                continue
            if (run / 'exports' / (old.name + '.pt')).exists() or checkpoint_iteration(old) in (2, 3):
                record('prune_old_resumable', path=str(old), retained=str(checkpoint))
                shutil.rmtree(old)
    def stage_summary(report):
        m = report['metrics']
        return {'slot1_mm_all': m['all']['xyz_mm']['slot1_mean'], 'slot1_mm_stationary': m['stationary'].get('xyz_mm', {}).get('slot1_mean'),
                'slot1_mm_moving': m['moving'].get('xyz_mm', {}).get('slot1_mean'), 'mean_valid_mm': m['all']['xyz_mm']['mean_valid_slots'],
                'per_task_slot1_mm': {task: (v['xyz_mm']['slot1_mean'] if v.get('samples') else None)
                                      for task, v in report.get('metrics_by_task', {}).items()},
                'language_probe_decision_accuracy': report.get('language_probe', {}).get('decision_accuracy'),
                'endpoint_mm': m['all']['xyz_mm']['endpoint_mean'], 'jaw_accuracy': m['all']['jaw']['accuracy_valid_slots'],
                'value_abs_error': m['all'].get('value_abs_error', {}).get('mean')}
    try:
        quota = scratch_headroom()
        if quota['free_bytes_lower_bound'] < 160_000_000_000:
            raise RuntimeError('Need 160 GB scratch headroom for retained exports and two in-flight full checkpoints')
        record('preflight_passed', quota=quota, initial_weights=str(initial), init_format=init['format'], horizon=HORIZON, tasks=len(TASKS),
               microbatch=identity['microbatch'], activation_checkpoint=identity['activation_checkpoint'], allocation_seconds=allocation)
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
            result = evaluate(exported, run / 'qualification_validation.json', stride=300, steps=5, parity=8,
                              parity_tolerance=QUALIFICATION_PARITY_TOLERANCE_MM)
            if not result.get('serving_parity_passed'):
                raise RuntimeError('Serving parity qualification failed')
            atomic_json(qualification, {'passed': True, 'checkpoint': str(checkpoint), 'resume_step': checkpoint_iteration(checkpoint),
                                       'report': str(run / 'qualification_validation.json')})
            prune_older(checkpoint)
        if args.qualify_only:
            record('qualification_complete', checkpoint=str(checkpoint))
            return
        reports = []
        for step in range(SAVE_EVERY, MAX_UPDATES + 1, SAVE_EVERY):
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
                    exported = export(run / 'checkpoints' / f'iter_{step:09d}')
            if time.time() >= stop_at and not (run / 'evaluation' / f'{exported.stem}_validation_stride{STAGE_EVAL["stride"]}.json').exists():
                record('stopped_for_allocation_budget', checkpoint=str(checkpoint), pending_evaluation=str(exported))
                return
            report = evaluate(exported, run / 'evaluation' / f'{exported.stem}_validation_stride{STAGE_EVAL["stride"]}.json', **STAGE_EVAL)
            record('validation_dense', step=step, **stage_summary(report))
            reports.append((report, exported))
        (best, best_path), jaw_floor_met = select_checkpoint(reports)
        selected = {'checkpoint': str(best_path.resolve()), 'rule': SELECTION_RULE, 'jaw_floor_met_by_any_export': jaw_floor_met,
                    'mean_valid_mm': best['metrics']['all']['xyz_mm']['mean_valid_slots'],
                    'slot1_mm': best['metrics']['all']['xyz_mm']['slot1_mean'], 'jaw_accuracy': best['metrics']['all']['jaw']['accuracy_valid_slots'],
                    'candidates': [{'checkpoint': str(p.resolve()), **stage_summary(r)} for r, p in reports],
                    'test_consulted_for_selection': False}
        selection_path = run / 'selection.json'
        if selection_path.exists() and json.loads(selection_path.read_text()) != selected:
            raise ValueError('A different checkpoint selection is already locked')
        atomic_json(selection_path, selected)
        final = evaluate(best_path, run / 'selected_validation.json', stride=FINAL_EVAL['stride'], steps=FINAL_EVAL['steps'],
                         also_steps=FINAL_EVAL['also_steps'], future=True, parity=200, probe_rows=FINAL_EVAL['probe_rows'])
        record('selected_validation', parity_passed=final.get('serving_parity_passed'), **stage_summary(final))
        test = evaluate(best_path, run / 'selected_test.json', split='test', stride=FINAL_EVAL['stride'], steps=FINAL_EVAL['steps'],
                        future=True, probe_rows=FINAL_EVAL['probe_rows'], selection=selection_path)
        record('completed', **{k: v for k, v in selected.items() if k != 'candidates'}, test=stage_summary(test))
    except BaseException as error:
        record('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
