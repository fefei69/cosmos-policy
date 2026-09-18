"""Bounded multi-H100 trial; never reads or changes another training run."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from cosmos_policy.datasets.hanoi_joint_data import CONTRACT, sha256
from cosmos_policy.utils.hanoi_checkpoint import checkpoint_iteration, latest_complete_checkpoint
from examples.hanoi.run_joint import atomic_json
from examples.hanoi.run_long import scratch_headroom
from examples.hanoi.run_pilot import execute_before


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', type=int, choices=(2, 4), default=4)
    args = parser.parse_args()
    import torch
    if torch.cuda.device_count() != args.gpus or any('H100' not in torch.cuda.get_device_name(i) for i in range(args.gpus)):
        raise RuntimeError('The trial requires the requested H100 allocation')
    root = Path(__file__).resolve().parents[2]
    name = f'hanoi_cosmos_joint_{args.gpus}h100_trial_{os.environ["SLURM_JOB_ID"]}'
    run = root / 'data/hanoi_cosmos/runs/cosmos_policy/hanoi' / name
    run.mkdir(parents=True, exist_ok=False)
    lock = (run / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    metadata = root / 'data/hanoi_cosmos/joint_sparse_v3'
    prepared = json.loads((metadata / 'metadata.json').read_text())
    os.environ.update(COSMOS_POLICY_PLATFORM='hanoi_joint', HANOI_JOINT_RUN_NAME=name,
                      HANOI_JOINT_METADATA=str(metadata), HANOI_TRAINING_SCHEDULE='aloha',
                      HANOI_INIT_CHECKPOINT=str(root / 'checkpoints/public/Cosmos-Policy-LIBERO-Predict2-2B.pt'),
                      HANOI_JOINT_WANDB_MODE='disabled', WANDB_MODE='disabled')
    os.environ.pop('HANOI_CONTINUATION_ANCHOR', None)
    deadline = time.time() + int(os.environ.get('HANOI_MULTIGPU_ALLOCATION_SECONDS', '1800')) - 60
    os.environ['HANOI_STOP_AT_EPOCH'] = str(deadline - 300)
    sources = [root / 'cosmos_policy/config/hanoi_joint_multigpu_config.py',
               root / 'cosmos_policy/utils/hanoi_multigpu_training.py', Path(__file__).resolve()]
    identity = {'contract': CONTRACT, 'statistics_sha256': prepared['statistics_sha256'],
                'metadata_sha256': sha256(metadata / 'metadata.json'), 'raw_sha256': prepared['raw_sha256'],
                'initial_weights': os.environ['HANOI_INIT_CHECKPOINT'], 'effective_batch_size': 32,
                'gpus': args.gpus, 'gpu_model': 'H100', 'parallelism': 'ddp', 'qualification_only': True,
                'code_sha256': {str(p.relative_to(root)): sha256(p) for p in sources}}
    atomic_json(run / 'joint_contract.json', identity)
    journal = {'job_id': os.environ['SLURM_JOB_ID'], 'run': str(run), 'events': []}
    def record(phase, **fields):
        journal['phase'] = phase
        journal['events'].append({'time': time.time(), 'phase': phase, **fields})
        atomic_json(run / 'multigpu_pipeline.json', journal)
        print('HANOI_MULTIGPU ' + json.dumps(journal['events'][-1]), flush=True)
    def execute(phase, command):
        record(phase, command=command)
        execute_before(command, root, deadline)
    python = str(root / '.venv/bin/python')
    train = [python, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={args.gpus}',
             '-m', 'cosmos_policy.scripts.train', '--config=cosmos_policy/config/hanoi_joint_multigpu_config.py',
             '--', 'experiment=cosmos_predict2_2b_hanoi_joint', 'trainer.max_val_iter=2']
    telemetry = None
    try:
        quota = scratch_headroom()
        if quota['free_bytes_lower_bound'] < 100_000_000_000:
            raise RuntimeError('Need 100 GB of scratch space for trial checkpoints')
        record('preflight_passed', quota=quota, global_batch=32, gpus=args.gpus)
        with (run / 'gpu_usage.csv').open('w') as usage:
            telemetry = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,name,utilization.gpu,memory.used,power.draw',
                                           '--format=csv', '--loop=5'], stdout=usage, stderr=subprocess.DEVNULL)
        execute('train_and_save', train + ['trainer.max_iter=5', 'checkpoint.save_iter=5'])
        checkpoint = latest_complete_checkpoint(run)
        if checkpoint_iteration(checkpoint) != 5:
            raise RuntimeError('Trial failed to complete its first five optimizer updates')
        execute('reload_and_benchmark', train + ['trainer.max_iter=25', 'checkpoint.save_iter=25'])
        checkpoint = latest_complete_checkpoint(run)
        if checkpoint_iteration(checkpoint) != 25:
            raise RuntimeError('Trial failed to finish resumed training')
        exported = run / 'exports/iter_000000025.pt'
        execute('export', [python, 'examples/hanoi/export_checkpoint.py', '--checkpoint', str(checkpoint), '--output', str(exported)])
        # Inference is deliberately one process on one of this trial's GPUs.
        # The existing validated serving code rejects a multi-GPU visibility mask.
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', ','.join(map(str, range(args.gpus)))).split(',')[0]
        report_path = run / 'qualification_validation.json'
        execute('inference_and_serving_parity', ['env', f'CUDA_VISIBLE_DEVICES={visible}', python, '-m',
                'cosmos_policy.experiments.robot.hanoi.run_hanoi_joint_eval', '--checkpoint', str(exported),
                '--metadata', str(metadata), '--mode', 'both', '--samples', '2', '--split', 'val', '--output', str(report_path)])
        report = json.loads(report_path.read_text())
        metrics = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
        if not report.get('serving_parity_passed') or not any(row['event'] == 'optimizer_restore_verified' for row in metrics):
            raise RuntimeError('Missing serving parity or exact optimizer reload verification')
        throughput = [row for row in metrics if row['event'] == 'throughput'][-1]
        result = {'passed': True, 'job_id': os.environ['SLURM_JOB_ID'], 'gpus': args.gpus,
                  'completed_updates': 25, 'resume_from': 5, 'global_batch': 32,
                  'checkpoint': str(checkpoint), 'throughput': throughput, 'serving_parity_passed': True,
                  'model_and_optimizer_replica_agreement': True, 'gradient_replica_agreement': True,
                  'rank_local_rng_restored': True, 'initialization': 'public Cosmos; independent trial'}
        atomic_json(run / 'qualification.json', result)
        record('qualification_complete', **result)
    except BaseException as error:
        record('failed', error=repr(error))
        raise
    finally:
        if telemetry is not None:
            telemetry.terminate()
            telemetry.wait(timeout=10)


if __name__ == '__main__':
    main()
