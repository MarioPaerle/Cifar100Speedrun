# Smoke Result

Job `48374534` ran on Leonardo under `IscrC_SIMP` on 2026-07-03.

Purpose: verify that the benchmark executes end-to-end. This is not target-selection or official baseline evidence.

Hardware: NVIDIA A100-SXM-64GB on `lrdn1122`.

Command path: `slurm/smoke.sh`, which runs one warmup and one tiny `C100_EPOCHS=0.05` run.

Observed output:

```text
config model=simple_resnet_muon runs=1 epochs=0.05 batch=1024 target=0.01 no_tta=1
|  warmup  |   eval  |     0.0242  |   0.0198  |       1.0000  |      29.4511  |
|       1  |   eval  |     0.0146  |   0.0143  |       1.0000  |       0.1315  |
```

Conclusion: data staging, model construction, Muon step, training loop, plain no-TTA validation, timer-before-validation behavior, and Slurm `IscrC_SIMP` account guard all execute.

Important: this is a smoke check only. It does not estimate the official 50-run baseline mean or justify changing the chosen target.
