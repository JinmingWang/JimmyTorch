"""Continuous score-based diffusion processes and predictor-corrector sampling."""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import log
from numbers import Integral, Real
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import torch
from tqdm import tqdm

Tensor = torch.Tensor
ContinuousTimestep = Union[float, Tensor]
ScoreFunction = Callable[..., Tensor]


class _ScoreSDE(ABC):
    """Shared continuous-time validation and perturbation utilities.

    Reference:
        Song et al., "Score-Based Generative Modeling through Stochastic
        Differential Equations," 2021. https://arxiv.org/abs/2011.13456
    """

    end_time = 1.0
    T = 1.0

    def __init__(self, device: str) -> None:
        self._torch_device = torch.empty(0, device=device).device
        self.device = str(self._torch_device)

    def _validateSample(self, samples: Tensor, name: str) -> None:
        if not isinstance(samples, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if samples.ndim < 1:
            raise ValueError(f"{name} must include a batch dimension")
        if not torch.is_floating_point(samples):
            raise TypeError(f"{name} must have a floating-point dtype")
        if samples.device != self._torch_device:
            raise ValueError(
                f"{name} is on {samples.device}, but the SDE is on "
                f"{self._torch_device}; construct the process on the sample device"
            )

    def _validateMatchingTensor(
        self, reference: Tensor, candidate: Tensor, name: str
    ) -> None:
        self._validateSample(candidate, name)
        if candidate.shape != reference.shape:
            raise ValueError(
                f"{name} shape {tuple(candidate.shape)} must match sample shape "
                f"{tuple(reference.shape)}"
            )
        if candidate.dtype != reference.dtype:
            raise TypeError(
                f"{name} dtype {candidate.dtype} must match sample dtype "
                f"{reference.dtype}"
            )

    def _normalizeTime(
        self,
        timestep: ContinuousTimestep,
        batch_size: int,
        dtype: torch.dtype,
        name: str,
    ) -> Tensor:
        if isinstance(timestep, Real) and not isinstance(timestep, bool):
            normalized_timestep = torch.tensor(
                float(timestep), device=self._torch_device, dtype=dtype
            )
        elif isinstance(timestep, Tensor):
            if timestep.device != self._torch_device:
                raise ValueError(
                    f"{name} is on {timestep.device}, but the SDE is on "
                    f"{self._torch_device}"
                )
            if not torch.is_floating_point(timestep):
                raise TypeError(f"{name} must have a floating-point dtype")
            if timestep.ndim > 1:
                raise ValueError(f"{name} must be a scalar or one-dimensional tensor")
            if timestep.ndim == 1 and timestep.numel() not in {1, batch_size}:
                raise ValueError(
                    f"{name} must contain one value or one value per batch item"
                )
            normalized_timestep = timestep.to(dtype=dtype)
        else:
            raise TypeError(f"{name} must be a real number or floating-point tensor")

        if not torch.all(torch.isfinite(normalized_timestep)).item():
            raise ValueError(f"{name} must contain only finite values")
        if torch.any((normalized_timestep < 0) | (normalized_timestep > 1)).item():
            raise ValueError(f"{name} must be in the continuous interval [0, 1]")
        return normalized_timestep

    @staticmethod
    def _reshapeCoefficient(coefficient: Tensor, samples: Tensor) -> Tensor:
        if coefficient.ndim == 0:
            return coefficient
        return coefficient.reshape(coefficient.shape[0], *((1,) * (samples.ndim - 1)))

    @staticmethod
    def _randnLike(samples: Tensor, generator: Optional[torch.Generator]) -> Tensor:
        return torch.randn(
            samples.shape,
            dtype=samples.dtype,
            device=samples.device,
            generator=generator,
        )

    @abstractmethod
    def driftDiffusion(
        self, x_t: Tensor, t: ContinuousTimestep
    ) -> Tuple[Tensor, Tensor]:
        """Return forward-SDE drift and diffusion at continuous time ``t``."""

    @abstractmethod
    def marginalProb(self, x_0: Tensor, t: ContinuousTimestep) -> Tuple[Tensor, Tensor]:
        """Return the mean and standard deviation of ``p_t(x_t | x_0)``."""

    @abstractmethod
    def priorSampling(
        self,
        shape: Sequence[int],
        *,
        generator: Optional[torch.Generator] = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Sample the terminal prior on the configured device."""

    @abstractmethod
    def _correctorScale(self, t: Tensor, num_steps: int) -> Tensor:
        """Return the SDE-specific scale used by the Langevin corrector."""

    def diffuse(
        self,
        x_0: Tensor,
        t: ContinuousTimestep,
        epsilon_0_to_t: Optional[Tensor] = None,
        *,
        generator: Optional[torch.Generator] = None,
        noise: Optional[Tensor] = None,
    ) -> Tensor:
        """Perturb clean samples with accumulated noise ``epsilon_0_to_t``."""
        self._validateSample(x_0, "x_0")
        if epsilon_0_to_t is not None and noise is not None:
            raise ValueError("provide only epsilon_0_to_t or its legacy alias noise")
        if epsilon_0_to_t is None:
            epsilon_0_to_t = noise
        if epsilon_0_to_t is None:
            epsilon_0_to_t = self._randnLike(x_0, generator)
        else:
            self._validateMatchingTensor(x_0, epsilon_0_to_t, "epsilon_0_to_t")

        marginal_mean, marginal_standard_deviation = self.marginalProb(x_0, t)
        broadcast_standard_deviation = self._reshapeCoefficient(
            marginal_standard_deviation, x_0
        )
        return marginal_mean + broadcast_standard_deviation * epsilon_0_to_t

    def computeScoreTarget(
        self,
        epsilon_0_to_t: Tensor,
        t: ContinuousTimestep,
    ) -> Tensor:
        """Return the denoising score target ``-epsilon_0_to_t / std(t)``."""
        self._validateSample(epsilon_0_to_t, "epsilon_0_to_t")
        zero_reference = torch.zeros_like(epsilon_0_to_t)
        _, marginal_standard_deviation = self.marginalProb(zero_reference, t)
        if torch.any(marginal_standard_deviation <= 0).item():
            raise ValueError("the score target is undefined at a zero-noise timestep")
        return -epsilon_0_to_t / self._reshapeCoefficient(
            marginal_standard_deviation, epsilon_0_to_t
        )

    def denoiseStep(
        self,
        x_t: Tensor,
        t: ContinuousTimestep,
        t_prev: ContinuousTimestep,
        score_func: ScoreFunction,
        need_mean: bool = False,
        *,
        corrector_steps: int = 1,
        snr: float = 0.16,
        num_steps: int = 1000,
        generator: Optional[torch.Generator] = None,
        **score_func_args: Any,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Run one predictor-corrector reverse transition from ``t`` to ``t_prev``."""
        sampler = PredictorCorrectorSampler(
            self,
            num_steps=num_steps,
            corrector_steps=corrector_steps,
            snr=snr,
            epsilon=1e-3,
        )
        return sampler.denoiseStep(
            x_t,
            t,
            t_prev,
            score_func,
            need_mean=need_mean,
            generator=generator,
            **score_func_args,
        )

    @torch.no_grad()
    def denoise(
        self,
        initial_noise: Tensor,
        score_func: ScoreFunction,
        verbose: bool = False,
        *,
        num_steps: int = 1000,
        corrector_steps: int = 1,
        snr: float = 0.16,
        epsilon: float = 1e-3,
        generator: Optional[torch.Generator] = None,
        **score_func_args: Any,
    ) -> Tensor:
        """Convenience wrapper around :class:`PredictorCorrectorSampler`."""
        sampler = PredictorCorrectorSampler(
            self,
            num_steps=num_steps,
            corrector_steps=corrector_steps,
            snr=snr,
            epsilon=epsilon,
        )
        return sampler.denoise(
            initial_noise,
            score_func,
            verbose=verbose,
            generator=generator,
            **score_func_args,
        )


class VPSDE(_ScoreSDE):
    """Variance-preserving continuous score-based diffusion process.

    Reference:
        Song et al., "Score-Based Generative Modeling through Stochastic
        Differential Equations," 2021. https://arxiv.org/abs/2011.13456
    """

    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        device: str = "cuda",
    ) -> None:
        if not isinstance(beta_min, Real) or isinstance(beta_min, bool):
            raise TypeError("beta_min must be a real number")
        if not isinstance(beta_max, Real) or isinstance(beta_max, bool):
            raise TypeError("beta_max must be a real number")
        if not 0 < float(beta_min) <= float(beta_max):
            raise ValueError("beta bounds must satisfy 0 < beta_min <= beta_max")
        super().__init__(device=device)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)

    def driftDiffusion(
        self, x_t: Tensor, t: ContinuousTimestep
    ) -> Tuple[Tensor, Tensor]:
        """Return ``(-0.5 * beta(t) * x_t, sqrt(beta(t)))``."""
        self._validateSample(x_t, "x_t")
        continuous_timesteps = self._normalizeTime(t, x_t.shape[0], x_t.dtype, "t")
        beta_t = self.beta_min + continuous_timesteps * (self.beta_max - self.beta_min)
        drift = -0.5 * self._reshapeCoefficient(beta_t, x_t) * x_t
        return drift, torch.sqrt(beta_t)

    def marginalProb(self, x_0: Tensor, t: ContinuousTimestep) -> Tuple[Tensor, Tensor]:
        """Return the analytic VP perturbation mean and standard deviation."""
        self._validateSample(x_0, "x_0")
        continuous_timesteps = self._normalizeTime(t, x_0.shape[0], x_0.dtype, "t")
        log_mean_coefficient = (
            -0.25 * continuous_timesteps.square() * (self.beta_max - self.beta_min)
            - 0.5 * continuous_timesteps * self.beta_min
        )
        mean_coefficient = torch.exp(log_mean_coefficient)
        marginal_mean = self._reshapeCoefficient(mean_coefficient, x_0) * x_0
        marginal_standard_deviation = torch.sqrt(
            (1.0 - torch.exp(2.0 * log_mean_coefficient)).clamp_min(0.0)
        )
        return marginal_mean, marginal_standard_deviation

    def priorSampling(
        self,
        shape: Sequence[int],
        *,
        generator: Optional[torch.Generator] = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Sample the standard-normal VP terminal prior."""
        normalized_shape = _validateShape(shape)
        _validateFloatingDtype(dtype)
        return torch.randn(
            normalized_shape,
            device=self._torch_device,
            dtype=dtype,
            generator=generator,
        )

    def _correctorScale(self, t: Tensor, num_steps: int) -> Tensor:
        beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
        return (1.0 - beta_t / num_steps).clamp_min(torch.finfo(t.dtype).eps)


class VESDE(_ScoreSDE):
    """Variance-exploding continuous score-based diffusion process.

    Reference:
        Song et al., "Score-Based Generative Modeling through Stochastic
        Differential Equations," 2021. https://arxiv.org/abs/2011.13456
    """

    def __init__(
        self,
        sigma_min: float = 0.01,
        sigma_max: float = 50.0,
        device: str = "cuda",
    ) -> None:
        if not isinstance(sigma_min, Real) or isinstance(sigma_min, bool):
            raise TypeError("sigma_min must be a real number")
        if not isinstance(sigma_max, Real) or isinstance(sigma_max, bool):
            raise TypeError("sigma_max must be a real number")
        if not 0 < float(sigma_min) < float(sigma_max):
            raise ValueError("sigma bounds must satisfy 0 < sigma_min < sigma_max")
        super().__init__(device=device)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self._log_sigma_ratio = log(self.sigma_max / self.sigma_min)

    def _sigma(self, t: Tensor) -> Tensor:
        return self.sigma_min * torch.exp(t * self._log_sigma_ratio)

    def driftDiffusion(
        self, x_t: Tensor, t: ContinuousTimestep
    ) -> Tuple[Tensor, Tensor]:
        """Return zero drift and the VE diffusion coefficient at ``t``."""
        self._validateSample(x_t, "x_t")
        continuous_timesteps = self._normalizeTime(t, x_t.shape[0], x_t.dtype, "t")
        sigma_t = self._sigma(continuous_timesteps)
        diffusion = sigma_t * (2.0 * self._log_sigma_ratio) ** 0.5
        return torch.zeros_like(x_t), diffusion

    def marginalProb(self, x_0: Tensor, t: ContinuousTimestep) -> Tuple[Tensor, Tensor]:
        """Return the analytic VE perturbation mean and standard deviation."""
        self._validateSample(x_0, "x_0")
        continuous_timesteps = self._normalizeTime(t, x_0.shape[0], x_0.dtype, "t")
        return x_0, self._sigma(continuous_timesteps)

    def priorSampling(
        self,
        shape: Sequence[int],
        *,
        generator: Optional[torch.Generator] = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Sample the Gaussian VE terminal prior with scale ``sigma_max``."""
        normalized_shape = _validateShape(shape)
        _validateFloatingDtype(dtype)
        return self.sigma_max * torch.randn(
            normalized_shape,
            device=self._torch_device,
            dtype=dtype,
            generator=generator,
        )

    def _correctorScale(self, t: Tensor, num_steps: int) -> Tensor:
        return torch.ones_like(t)


class PredictorCorrectorSampler:
    """Euler-Maruyama predictor with an optional Langevin corrector.

    ``score_func`` must accept ``(noisy_samples, continuous_timesteps,
    **kwargs)`` and return a tensor matching the sample. Setting
    ``corrector_steps=0`` selects predictor-only sampling. The constructor's
    scalar ``epsilon`` is a terminal time, not a noise tensor; perturbation
    noise is named ``epsilon_0_to_t`` throughout the process API.

    Reference:
        Song et al., "Score-Based Generative Modeling through Stochastic
        Differential Equations," 2021. https://arxiv.org/abs/2011.13456
    """

    def __init__(
        self,
        sde: _ScoreSDE,
        num_steps: int = 1000,
        corrector_steps: int = 1,
        snr: float = 0.16,
        epsilon: float = 1e-3,
    ) -> None:
        if not isinstance(sde, _ScoreSDE):
            raise TypeError("sde must be a VPSDE or VESDE instance")
        if not isinstance(num_steps, Integral) or isinstance(num_steps, bool):
            raise TypeError("num_steps must be an integer")
        if num_steps < 1:
            raise ValueError("num_steps must be at least 1")
        if not isinstance(corrector_steps, Integral) or isinstance(
            corrector_steps, bool
        ):
            raise TypeError("corrector_steps must be an integer")
        if corrector_steps < 0:
            raise ValueError("corrector_steps must be non-negative")
        if not isinstance(snr, Real) or isinstance(snr, bool):
            raise TypeError("snr must be a real number")
        if float(snr) <= 0:
            raise ValueError("snr must be positive")
        if not isinstance(epsilon, Real) or isinstance(epsilon, bool):
            raise TypeError("epsilon must be a real number")
        if not 0 < float(epsilon) < 1:
            raise ValueError("epsilon must satisfy 0 < epsilon < 1")

        self.sde = sde
        self.num_steps = int(num_steps)
        self.corrector_steps = int(corrector_steps)
        self.snr = float(snr)
        self.epsilon = float(epsilon)
        self.device = sde.device

    def _evaluateScore(
        self,
        samples: Tensor,
        continuous_timesteps: Tensor,
        score_func: ScoreFunction,
        score_func_args: dict[str, Any],
    ) -> Tensor:
        score = score_func(samples, continuous_timesteps, **score_func_args)
        if not isinstance(score, Tensor):
            raise TypeError("score_func must return a torch.Tensor")
        self.sde._validateMatchingTensor(samples, score, "score")
        return score

    def _correct(
        self,
        samples: Tensor,
        batch_timesteps: Tensor,
        score_func: ScoreFunction,
        generator: Optional[torch.Generator],
        score_func_args: dict[str, Any],
    ) -> Tensor:
        corrected_samples = samples
        for _ in range(self.corrector_steps):
            score = self._evaluateScore(
                corrected_samples,
                batch_timesteps,
                score_func,
                score_func_args,
            )
            langevin_noise = self.sde._randnLike(corrected_samples, generator)
            score_norm = torch.linalg.vector_norm(
                score.reshape(score.shape[0], -1), dim=1
            ).mean()
            noise_norm = torch.linalg.vector_norm(
                langevin_noise.reshape(langevin_noise.shape[0], -1), dim=1
            ).mean()
            minimum_norm = torch.finfo(corrected_samples.dtype).eps
            safe_score_norm = score_norm.clamp_min(minimum_norm)
            scale = self.sde._correctorScale(batch_timesteps, self.num_steps)
            adaptive_step_size = torch.where(
                score_norm > minimum_norm,
                2.0 * scale * (self.snr * noise_norm / safe_score_norm).square(),
                torch.zeros_like(scale),
            )
            broadcast_step_size = self.sde._reshapeCoefficient(
                adaptive_step_size, corrected_samples
            )
            corrected_mean = corrected_samples + broadcast_step_size * score
            corrected_samples = (
                corrected_mean + torch.sqrt(2.0 * broadcast_step_size) * langevin_noise
            )
        return corrected_samples

    def denoiseStep(
        self,
        x_t: Tensor,
        t: ContinuousTimestep,
        t_prev: ContinuousTimestep,
        score_func: ScoreFunction,
        need_mean: bool = False,
        *,
        generator: Optional[torch.Generator] = None,
        **score_func_args: Any,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Run one Langevin-corrected Euler-Maruyama reverse step."""
        self.sde._validateSample(x_t, "x_t")
        batch_size = x_t.shape[0]
        current_timesteps = self.sde._normalizeTime(t, batch_size, x_t.dtype, "t")
        previous_timesteps = self.sde._normalizeTime(
            t_prev, batch_size, x_t.dtype, "t_prev"
        )
        if not torch.all(previous_timesteps < current_timesteps).item():
            raise ValueError("t_prev must be strictly smaller than t")
        batch_timesteps = current_timesteps.expand(batch_size)
        batch_previous_timesteps = previous_timesteps.expand(batch_size)

        corrected_samples = self._correct(
            x_t,
            batch_timesteps,
            score_func,
            generator,
            score_func_args,
        )
        score = self._evaluateScore(
            corrected_samples,
            batch_timesteps,
            score_func,
            score_func_args,
        )
        forward_drift, forward_diffusion = self.sde.driftDiffusion(
            corrected_samples, batch_timesteps
        )
        broadcast_diffusion = self.sde._reshapeCoefficient(
            forward_diffusion, corrected_samples
        )
        reverse_drift = forward_drift - broadcast_diffusion.square() * score
        time_delta = batch_previous_timesteps - batch_timesteps
        broadcast_time_delta = self.sde._reshapeCoefficient(
            time_delta, corrected_samples
        )
        previous_mean = corrected_samples + reverse_drift * broadcast_time_delta
        predictor_noise = self.sde._randnLike(corrected_samples, generator)
        previous_samples = (
            previous_mean
            + broadcast_diffusion * torch.sqrt(-broadcast_time_delta) * predictor_noise
        )

        if need_mean:
            return previous_samples, previous_mean
        return previous_samples

    @torch.no_grad()
    def denoise(
        self,
        initial_noise: Tensor,
        score_func: ScoreFunction,
        verbose: bool = False,
        *,
        generator: Optional[torch.Generator] = None,
        **score_func_args: Any,
    ) -> Tensor:
        """Integrate the reverse SDE from time ``1`` to ``epsilon``."""
        self.sde._validateSample(initial_noise, "initial_noise")
        current_samples = initial_noise.clone()
        sampling_schedule = torch.linspace(
            1.0,
            self.epsilon,
            self.num_steps + 1,
            device=current_samples.device,
            dtype=current_samples.dtype,
        )
        sampling_pairs = list(zip(sampling_schedule[:-1], sampling_schedule[1:]))
        sampling_iterator = (
            tqdm(sampling_pairs, desc="Score-SDE sampling")
            if verbose
            else sampling_pairs
        )

        final_mean = current_samples
        for current_timestep, previous_timestep in sampling_iterator:
            current_samples, final_mean = self.denoiseStep(
                current_samples,
                current_timestep,
                previous_timestep,
                score_func,
                need_mean=True,
                generator=generator,
                **score_func_args,
            )
        return final_mean


def _validateShape(shape: Sequence[int]) -> Tuple[int, ...]:
    if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)):
        raise TypeError("shape must be a sequence of positive integers")
    normalized_shape = tuple(shape)
    if not normalized_shape:
        raise ValueError("shape must include a batch dimension")
    if any(
        not isinstance(dimension, Integral)
        or isinstance(dimension, bool)
        or dimension < 1
        for dimension in normalized_shape
    ):
        raise ValueError("shape must contain only positive integers")
    return normalized_shape


def _validateFloatingDtype(dtype: torch.dtype) -> None:
    if (
        not isinstance(dtype, torch.dtype)
        or not torch.empty((), dtype=dtype).is_floating_point()
    ):
        raise TypeError("dtype must be a floating-point torch.dtype")
