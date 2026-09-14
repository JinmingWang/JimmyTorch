"""Non-gating discrete diffusion sampling microbenchmark.

Run from the repository root with::

    python benchmarks/benchmark_diffusion.py
"""

import argparse
from pathlib import Path
import statistics
import sys
import time

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from DiffusionModels import DDIM, DDPM, DPMSolverPP


def benchmark(process, initial_noise, repeats):
    def predict(samples, batch_timesteps):
        del batch_timesteps
        return None, torch.zeros_like(samples), None

    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        process.denoise(initial_noise, predict)
        if initial_noise.device.type == "cuda":
            torch.cuda.synchronize(initial_noise.device)
        durations.append(time.perf_counter() - started)
    return statistics.median(durations)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--skip-step", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--length", type=int, default=128)
    args = parser.parse_args()

    initial_noise = torch.randn(
        args.batch_size,
        args.channels,
        args.length,
        device=args.device,
    )
    ddim = DDIM(
        max_diffusion_step=args.steps,
        device=args.device,
        skip_step=args.skip_step,
    )
    ddim_schedule = list(range(args.steps - 1, -1, -args.skip_step))
    if ddim_schedule[-1] != 0:
        ddim_schedule.append(0)
    processes = (
        DDPM(max_diffusion_step=args.steps, device=args.device),
        ddim,
        DPMSolverPP(
            max_diffusion_step=args.steps,
            device=args.device,
            num_inference_steps=len(ddim_schedule),
            solver_order=2,
        ),
    )
    for process in processes:
        if initial_noise.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(initial_noise.device)
        median_duration = benchmark(process, initial_noise, args.repeats)
        memory_suffix = ""
        if initial_noise.device.type == "cuda":
            peak_memory = torch.cuda.max_memory_allocated(initial_noise.device)
            memory_suffix = f", peak_memory={peak_memory / 1024**2:.2f} MiB"
        print(
            f"{type(process).__name__}: median={median_duration:.6f}s"
            f"{memory_suffix}"
        )


if __name__ == "__main__":
    main()
