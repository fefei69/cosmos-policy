# Four-H200 diagnostic

Job **17863565** was submitted on September 15, 2026 at 14:48:51 EDT under
`torch_pr_595_tandon_advanced`. It requests four H200 GPUs on one node, eight
CPUs, 32 GiB host memory, and a maximum of ten minutes. Slurm selects the
permitted partitions automatically. There are no retries or requeues.

## Completed result

The job **passed** on node **gh126**, running from **18:10:46 to 18:11:28 EDT**
on September 15, 2026. Slurm recorded `COMPLETED`, exit code `0:0`, and an
elapsed time of **42 seconds**. The queue wait was **3 hours 21 minutes 55 seconds**.
The JSON report verifies four distinct H200 devices, passing BF16 arithmetic,
all three NCCL collective checks, and three synchronized diagnostic updates.
The allocation was released when the job completed.

This confirms that the account successfully obtained and used four H200s on
one node with the current Python/CUDA environment. Future queue times can differ.

The first queue check showed `PENDING / QOSGrpGRES`: waiting for capacity under
the partition's shared GPU limit. The submitted job had no estimated start time.
The earlier test-only estimate is not a reservation.

## What it checks

- Four H200 CUDA devices and four worker processes on one node.
- A BF16 matrix multiplication with a known correct answer on each GPU.
- NCCL sums across all four GPUs for three tensor sizes.
- Three optimizer updates of a small distributed diagnostic model, followed
  by a check that all workers have identical finite weights.

It does not access Hanoi recordings, Cosmos weights, or training checkpoints.
It does not establish Cosmos training speed, policy quality on H200, checkpoint
migration correctness, or guaranteed availability for future jobs.

## Results and reuse

- Submission audit: `data/hanoi_cosmos/operations/four_h200_check_submission_20260915.json`
- Success report: `data/hanoi_cosmos/operations/four_h200_check_17863565.json`
- Logs: `data/hanoi_cosmos/logs/cosmos-4h200-check-17863565.out` and `.err`

Require both a passing report and Slurm `COMPLETED` with exit code `0:0`.
A queued job alone does not mean the GPU tests have passed.

For a future authorized diagnostic, submit from the project root:

```bash
sbatch --parsable examples/hanoi/check_four_h200.sbatch
```

The script records a new report using the allocated job ID, then exits and
releases its resources. Cosmos training job **17851770** continues on one H100.
