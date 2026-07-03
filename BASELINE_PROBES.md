# Baseline Probes

These are setup probes, not official 30-run benchmark results.

## Eager 12 Epoch Probe

Job `48375215`, `IscrC_SIMP`, A100-SXM-64GB on `lrdn3443`.

```text
config model=simple_resnet_muon runs=1 epochs=12.0 batch=1024 target=0.6 no_tta=1
|  warmup  |   eval  |     0.2760  |   0.2657  |       0.0000  |      26.7581  |
|       1  |   eval  |     0.8812  |   0.6970  |       1.0000  |      24.9036  |
```

Interpretation: the `69.70%` validation result is real plain no-TTA validation, but it was eager and single-seed. It is not enough margin for a `70%` official threshold.

## Max-Autotune Compile Attempt

Job `48375667` used `C100_COMPILE_MODE=max-autotune` and was canceled after `4:05` because it was still autotuning. Logs showed multiple convolution/MM autotune blocks, including one `25.56s` block. This mode is intentionally not the default.

## Reduce-Overhead Compile Smoke

Job `48376039`, `IscrC_SIMP`, A100-SXM-64GB on `lrdn0071`.

```text
config model=simple_resnet_muon runs=1 epochs=0.05 batch=1024 target=0.01 compile=1 compile_mode=reduce-overhead no_tta=1
|  warmup  |   eval  |     0.0128  |   0.0118  |       1.0000  |      50.3267  |
|       1  |   eval  |     0.0123  |   0.0121  |       1.0000  |       0.1038  |
```

Interpretation: lightweight compile works. The compile/cold-start cost is around `40-50s` on the first warmup, then measured runs are fast.

## Compiled 14 Epoch Probe

Job `48376210`, `IscrC_SIMP`, A100-SXM-64GB on `lrdn0746`.

```text
config model=simple_resnet_muon runs=1 epochs=14.0 batch=1024 target=0.7 compile=1 compile_mode=reduce-overhead no_tta=1
|  warmup  |   eval  |     0.2759  |   0.2736  |       0.0000  |      44.9931  |
|       1  |   eval  |     0.9098  |   0.7011  |       1.0000  |      22.8165  |
```

Interpretation: real, but too close to `70%` for the official benchmark.

## Compiled 16 Epoch Probe

Job `48376444`, `IscrC_SIMP`, A100-SXM-64GB on `lrdn0122`.

```text
config model=simple_resnet_muon runs=1 epochs=16.0 batch=1024 target=0.7 compile=1 compile_mode=reduce-overhead no_tta=1
|  warmup  |   eval  |     0.2752  |   0.2724  |       0.0000  |      38.9707  |
|       1  |   eval  |     0.9383  |   0.7058  |       1.0000  |      25.8564  |
```

Interpretation: this is the current v0 baseline setting for `k = 70%`: 16 epochs, compiled with `reduce-overhead`, target cleared on one seed. The official 30-run baseline is still unrun.

## Time Estimate

Measured one-run training time at 16 epochs is `25.86s`. Official 30-run timed training is therefore about `12.9 min`. Add `~40-50s` compile/cold warmup, sleeps, validation gates, and Slurm overhead; expected wall time is roughly `15-17 min` on a single Leonardo A100.
