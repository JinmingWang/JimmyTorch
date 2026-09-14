# Discrete diffusion sampling benchmark

Run the non-gating CPU or CUDA microbenchmark from the repository root:

```bash
python benchmarks/benchmark_diffusion.py --device cpu --repeats 10
```

The benchmark uses a `(32, 4, 128)` tensor, 100 diffusion steps, and a DDIM
skip step of 5 by default. DPM-Solver++ is assigned the same number of model
evaluations as DDIM. The script reports median denoising time and, on CUDA,
peak allocated memory.

Reference CPU measurements captured on 2026-09-14 with PyTorch 2.5.1:

| Implementation | DDPM | DDIM | DPM-Solver++ (order 2) |
|---|---:|---:|---:|
| Before the shared discrete foundation | 0.009612 s | 0.000825 s | — |
| Increment 1 implementation | 0.008241 s | 0.000747 s | — |
| Current implementation | 0.007919 s | 0.000705 s | 0.001116 s |

These timings are informational rather than test thresholds. The refactor
removes sample flattening and schedule device transfers from the reverse loop;
formula and compatibility tests are the gating checks.
