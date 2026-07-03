# CIFAR-100 A100 Speedrun Autoresearch Benchmark

Remote path: `/leonardo_work/IscrC_YENDRI/paerle/Cifar100Speedrun`.

Goal: train on official CIFAR-100 train images and reach a fixed plain validation accuracy target `k = 70%` on a single A100 in the least training time possible.

Inspired by Keller Jordan CIFAR-10 Airbench and the local Leonardo CIFAR-10 replication, but validation is stricter: no TTA, no TTT, no confidence-triggered evaluation path, no ensembling, no validation-time adaptation, no calibration on validation labels.

## Chosen constants

- Target: `k = 70%` plain validation accuracy.
- Official run count: 30 runs.
- Official epoch budget: 16 epochs.
- Baseline: `train_cifar100_resnet_muon.py` only.
- Compiled baseline: enabled by default with `torch.compile` / `C100_COMPILE=1`, default `C100_COMPILE_MODE=reduce-overhead`; warmup pays compile/cold-start cost before measured runs. `max-autotune` is intentionally not the default because it can spend minutes autotuning on Leonardo.
- Timed quantity: training time only; validation stays frozen and untimed.

## Record metric

Every record must report both:

1. Absolute score: mean `time_seconds` over 30 official runs while clearing `mean(val_acc) > k`, where `k = 70%` plain validation accuracy.
2. Relative score: paired same-pod comparison against a replication of the baseline or last record, with the same seed/run list, reporting time ratio and delta.

A claim without the relative same-pod replication is not a record. This protects against A100, driver, node, clock, and thermal differences.

## Target and runs

Chosen target: `k = 70%` plain validation accuracy.

Chosen official run count: 30 runs. Fast triage may use 40 runs, smoke checks use 1 run, and 200 runs are reserved only for a final public artifact if 30-run uncertainty is disputed.

`slurm/discovery.sh` is kept as optional infrastructure for future calibration, but it is not part of the setup result and was not used to choose the v0 target.

For CIFAR-100 std around 0.4-0.6 percentage points, 30 runs gives SE around 0.06-0.085 percentage points. A true 0.2 percentage point target margin is useful; below 0.1 is fragile.

## Files

- `cifar100-benchmark/train_cifar100_resnet_muon.py`: default and only baseline, a deliberately simple PyTorch ResNet trained with Muon and compiled by default. This is the benchmark substrate.
- `cifar100-benchmark/prepare_cifar100.py`: downloads and packs CIFAR-100 into `train.pt` and `test.pt`.
- `cifar100-benchmark/analyze_cifar100.py`: parses benchmark logs and reports mean accuracy, time, and p-value approximation.
- `slurm/smoke.sh`: one tiny run to verify the benchmark executes; not evidence for target choice.
- `slurm/discovery.sh`: target discovery, not run during setup.
- `slurm/official_baseline.sh`: 30-run official baseline for `k = 70%`.

## Commands

Use Cineca account `IscrC_SIMP`. The Slurm scripts refuse to run outside `IscrC_SIMP`.


```bash
cd /leonardo_work/IscrC_YENDRI/paerle/Cifar100Speedrun
source env_setup.sh
python prepare_cifar100.py
sbatch slurm/smoke.sh
# Optional future calibration only: sbatch slurm/discovery.sh
```

## Hard validation rules

- Train split only for training.
- Official CIFAR-100 test split is the fixed validation set. The validation implementation must not be touched for records.
- No validation images or labels in optimizer state, schedules, data selection, augmentation selection, or per-example control flow.
- One plain forward pass for validation. No flips, crops, averaging, confidence branches, BN adaptation, EMA selection, or ensembles.
- Timing excludes validation. The timer stops before validation starts; validation is an untimed pass/fail gate.

## Evidence

See `BASELINE_PROBES.md` for smoke/probe logs. Current v0 setting is `k = 70%`, 30 official runs, 16 epochs, `C100_COMPILE_MODE=reduce-overhead`. The official 30-run baseline is still unrun.

## Feasibility note

The `k = 70%` target is mechanically configured and has one compiled 16-epoch seed clearing it at `70.58%`, but it is not yet validated over the official 30-run baseline. The smoke check only proves the code path executes. Run `slurm/official_baseline.sh` to measure whether the baseline clears 70% over 30 runs.

## Compiled one-seed probes

- Eager 12-epoch probe `48375215`: `69.70%` validation, `24.90s` timed training. Real plain validation, but not compiled and not enough margin for a 70% target.
- Compiled 14-epoch probe `48376210`: `70.11%` validation, `22.82s` timed training, warmup/compile row `44.99s`. Real but too close to 70 for a 30-run benchmark.
