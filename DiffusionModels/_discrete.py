"""Shared discrete Gaussian diffusion utilities.

The public process classes intentionally keep their historical camelCase API.
This private module centralizes the schedule and tensor algebra so that the
DDPM and DDIM implementations cannot silently drift apart.
"""

from __future__ import annotations

from math import log, pi
from numbers import Integral, Real
from typing import Literal, Optional, Tuple, Union

import torch

Tensor = torch.Tensor
Timestep = Union[int, Tensor]
ScaleMode = Literal["linear", "quadratic", "log", "cosine"]


class _DiscreteGaussianDiffusion:
    """Common schedule and prediction algebra for discrete diffusion.

    References:
        Ho, Jain, and Abbeel, "Denoising Diffusion Probabilistic Models,"
        2020. https://arxiv.org/abs/2006.11239
        Nichol and Dhariwal, "Improved Denoising Diffusion Probabilistic
        Models," 2021. https://arxiv.org/abs/2102.09672
    """

    clean_timestep = -1

    def __init__(
        self,
        min_beta: float,
        max_beta: float,
        max_diffusion_step: int,
        device: str,
        scale_mode: ScaleMode,
    ) -> None:
        self._validateConfiguration(
            min_beta=min_beta,
            max_beta=max_beta,
            max_diffusion_step=max_diffusion_step,
            scale_mode=scale_mode,
        )

        torch_device = torch.device(device)
        betas = self._buildBetas(
            min_beta=float(min_beta),
            max_beta=float(max_beta),
            max_diffusion_step=max_diffusion_step,
            device=torch_device,
            scale_mode=scale_mode,
        )
        # Resolve an unspecified CUDA index ("cuda") to the actual tensor device.
        torch_device = betas.device
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        previous_alpha_bars = torch.cat(
            [torch.ones(1, device=torch_device, dtype=betas.dtype), alpha_bars[:-1]]
        )

        posterior_variance = (
            betas * (1.0 - previous_alpha_bars) / (1.0 - alpha_bars)
        ).clamp_min(0.0)
        posterior_mean_x0_coefficients = (
            betas * torch.sqrt(previous_alpha_bars) / (1.0 - alpha_bars)
        )
        posterior_mean_xt_coefficients = (
            torch.sqrt(alphas) * (1.0 - previous_alpha_bars) / (1.0 - alpha_bars)
        )

        self.T = max_diffusion_step
        self.device = str(torch_device)
        self.scale_mode = scale_mode
        self.min_beta = float(min_beta)
        self.max_beta = float(max_beta)
        self._torch_device = torch_device

        self._base_schedules = {
            "betas": betas,
            "alphas": alphas,
            "alpha_bars": alpha_bars,
            "sqrt_alphas": torch.sqrt(alphas),
            "sqrt_one_minus_alphas": torch.sqrt(1.0 - alphas),
            "sqrt_alpha_bars": torch.sqrt(alpha_bars),
            "sqrt_one_minus_alpha_bars": torch.sqrt(1.0 - alpha_bars),
            "posterior_variance": posterior_variance,
            "posterior_mean_x0_coefficients": posterior_mean_x0_coefficients,
            "posterior_mean_xt_coefficients": posterior_mean_xt_coefficients,
        }
        self._schedule_cache = {betas.dtype: self._base_schedules}

        # Preserve the historical public (T, 1) schedule tensor shapes.
        self.betas = betas.view(-1, 1)
        self.alphas = alphas.view(-1, 1)
        self.alpha_bars = alpha_bars.view(-1, 1)
        self.sqrt_alpha_bars = self._base_schedules["sqrt_alpha_bars"].view(-1, 1)
        self.sqrt_1m_alpha_bars = self._base_schedules[
            "sqrt_one_minus_alpha_bars"
        ].view(-1, 1)
        self.posterior_variance = posterior_variance.view(-1, 1)
        self.posterior_mean_x0_coefficients = posterior_mean_x0_coefficients.view(-1, 1)
        self.posterior_mean_xt_coefficients = posterior_mean_xt_coefficients.view(-1, 1)

    @staticmethod
    def _validateConfiguration(
        min_beta: float,
        max_beta: float,
        max_diffusion_step: int,
        scale_mode: str,
    ) -> None:
        if not isinstance(min_beta, Real) or isinstance(min_beta, bool):
            raise TypeError("min_beta must be a real number")
        if not isinstance(max_beta, Real) or isinstance(max_beta, bool):
            raise TypeError("max_beta must be a real number")
        if not 0.0 < float(min_beta) <= float(max_beta) < 1.0:
            raise ValueError("beta bounds must satisfy 0 < min_beta <= max_beta < 1")
        if not isinstance(max_diffusion_step, Integral) or isinstance(
            max_diffusion_step, bool
        ):
            raise TypeError("max_diffusion_step must be an integer")
        if max_diffusion_step < 1:
            raise ValueError("max_diffusion_step must be at least 1")
        if scale_mode not in {"linear", "quadratic", "log", "cosine"}:
            raise ValueError(
                "scale_mode must be one of: linear, quadratic, log, cosine"
            )

    @staticmethod
    def _buildBetas(
        min_beta: float,
        max_beta: float,
        max_diffusion_step: int,
        device: torch.device,
        scale_mode: ScaleMode,
    ) -> Tensor:
        if scale_mode == "quadratic":
            betas = torch.linspace(
                min_beta**0.5,
                max_beta**0.5,
                max_diffusion_step,
                device=device,
                dtype=torch.float64,
            ).square()
        elif scale_mode == "log":
            betas = torch.exp(
                torch.linspace(
                    log(min_beta),
                    log(max_beta),
                    max_diffusion_step,
                    device=device,
                    dtype=torch.float64,
                )
            )
        elif scale_mode == "cosine":
            schedule_points = torch.arange(
                max_diffusion_step + 1, device=device, dtype=torch.float64
            )
            normalized_time = schedule_points / max_diffusion_step
            alpha_bars = torch.cos(
                ((normalized_time + 0.008) / 1.008) * pi / 2.0
            ).square()
            alpha_bars = alpha_bars / alpha_bars[0]
            betas = 1.0 - alpha_bars[1:] / alpha_bars[:-1]
            betas = betas.clamp(min=1e-12, max=0.999)
        else:
            betas = torch.linspace(
                min_beta,
                max_beta,
                max_diffusion_step,
                device=device,
                dtype=torch.float64,
            )
        return betas.to(dtype=torch.float32)

    def _validateSample(self, samples: Tensor, name: str) -> None:
        if not isinstance(samples, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if samples.ndim < 1:
            raise ValueError(f"{name} must include a batch dimension")
        if not torch.is_floating_point(samples):
            raise TypeError(f"{name} must have a floating-point dtype")
        if samples.device != self._torch_device:
            raise ValueError(
                f"{name} is on {samples.device}, but the diffusion schedule is on "
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
                f"{name} dtype {candidate.dtype} must match sample dtype {reference.dtype}"
            )

    def _normalizeTimestep(
        self,
        timestep: Timestep,
        batch_size: int,
        name: str,
        allow_clean: bool = False,
    ) -> Tensor:
        if isinstance(timestep, Integral) and not isinstance(timestep, bool):
            normalized_timestep = torch.tensor(
                int(timestep), device=self._torch_device, dtype=torch.long
            )
        elif isinstance(timestep, Tensor):
            if timestep.device != self._torch_device:
                raise ValueError(
                    f"{name} is on {timestep.device}, but the schedule is on "
                    f"{self._torch_device}"
                )
            if timestep.dtype not in {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            }:
                raise TypeError(f"{name} must use an integer dtype")
            if timestep.ndim > 1:
                raise ValueError(f"{name} must be a scalar or a one-dimensional tensor")
            if timestep.ndim == 1 and timestep.numel() not in {1, batch_size}:
                raise ValueError(
                    f"{name} must contain one value or one value per batch item"
                )
            normalized_timestep = timestep.to(dtype=torch.long)
        else:
            raise TypeError(f"{name} must be an int or integer tensor")

        minimum_timestep = self.clean_timestep if allow_clean else 0
        invalid = (normalized_timestep < minimum_timestep) | (
            normalized_timestep >= self.T
        )
        if torch.any(invalid).item():
            endpoint_note = f" or {self.clean_timestep}" if allow_clean else ""
            raise ValueError(f"{name} must be in [0, {self.T - 1}]{endpoint_note}")
        return normalized_timestep

    def _scheduleForDtype(self, dtype: torch.dtype) -> dict[str, Tensor]:
        cached_schedules = self._schedule_cache.get(dtype)
        if cached_schedules is None:
            cached_schedules = {
                name: values.to(dtype=dtype)
                for name, values in self._base_schedules.items()
            }
            self._schedule_cache[dtype] = cached_schedules
        return cached_schedules

    @staticmethod
    def _reshapeCoefficient(coefficient: Tensor, samples: Tensor) -> Tensor:
        if coefficient.ndim == 0:
            return coefficient
        return coefficient.reshape(coefficient.shape[0], *((1,) * (samples.ndim - 1)))

    def _extractCoefficient(
        self,
        name: str,
        timestep: Timestep,
        samples: Tensor,
        allow_clean: bool = False,
    ) -> Tensor:
        normalized_timestep = self._normalizeTimestep(
            timestep,
            batch_size=samples.shape[0],
            name="timestep",
            allow_clean=allow_clean,
        )
        return self._extractNormalizedCoefficient(
            name=name,
            normalized_timestep=normalized_timestep,
            samples=samples,
            allow_clean=allow_clean,
        )

    def _extractNormalizedCoefficient(
        self,
        name: str,
        normalized_timestep: Tensor,
        samples: Tensor,
        allow_clean: bool = False,
    ) -> Tensor:
        schedule = self._scheduleForDtype(samples.dtype)[name]
        if allow_clean:
            clean_mask = normalized_timestep == self.clean_timestep
            schedule_indices = normalized_timestep.clamp_min(0)
            coefficient = schedule[schedule_indices]
            coefficient = torch.where(
                clean_mask, torch.ones_like(coefficient), coefficient
            )
        else:
            coefficient = schedule[normalized_timestep]
        return self._reshapeCoefficient(coefficient, samples)

    def _validateTransition(
        self,
        current_timestep: Timestep,
        previous_timestep: Timestep,
        batch_size: int,
        adjacent_only: bool,
    ) -> Tuple[Tensor, Tensor]:
        normalized_current = self._normalizeTimestep(
            current_timestep, batch_size, "t", allow_clean=False
        )
        normalized_previous = self._normalizeTimestep(
            previous_timestep, batch_size, "t_prev", allow_clean=True
        )
        if adjacent_only:
            valid_transition = normalized_previous == normalized_current - 1
        else:
            valid_transition = normalized_previous < normalized_current
        if not torch.all(valid_transition).item():
            transition_type = "adjacent" if adjacent_only else "strictly descending"
            raise ValueError(
                f"t_prev must define a {transition_type} transition from t"
            )
        return normalized_current, normalized_previous

    def _parsePrediction(
        self,
        x_t: Tensor,
        t: Timestep,
        x0_pred: Optional[Tensor] = None,
        epsilon_0_to_t_pred: Optional[Tensor] = None,
        v_pred: Optional[Tensor] = None,
        timestep_is_normalized: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        predictions = {
            "x0_pred": x0_pred,
            "epsilon_0_to_t_pred": epsilon_0_to_t_pred,
            "v_pred": v_pred,
        }
        supplied_predictions = [
            name for name, prediction in predictions.items() if prediction is not None
        ]
        if len(supplied_predictions) != 1:
            raise ValueError(
                "exactly one of x0_pred, epsilon_0_to_t_pred "
                "(legacy name: epsilon_pred), or v_pred must be provided"
            )
        for name in supplied_predictions:
            supplied_prediction = predictions[name]
            assert supplied_prediction is not None
            self._validateMatchingTensor(x_t, supplied_prediction, name)

        if timestep_is_normalized:
            assert isinstance(t, Tensor)
            sqrt_alpha_bar = self._extractNormalizedCoefficient(
                "sqrt_alpha_bars", t, x_t
            )
            sqrt_one_minus_alpha_bar = self._extractNormalizedCoefficient(
                "sqrt_one_minus_alpha_bars", t, x_t
            )
        else:
            sqrt_alpha_bar = self._extractCoefficient("sqrt_alpha_bars", t, x_t)
            sqrt_one_minus_alpha_bar = self._extractCoefficient(
                "sqrt_one_minus_alpha_bars", t, x_t
            )

        if v_pred is not None:
            parsed_x0 = sqrt_alpha_bar * x_t - sqrt_one_minus_alpha_bar * v_pred
            parsed_epsilon_0_to_t = (
                sqrt_one_minus_alpha_bar * x_t + sqrt_alpha_bar * v_pred
            )
        elif epsilon_0_to_t_pred is not None:
            parsed_epsilon_0_to_t = epsilon_0_to_t_pred
            parsed_x0 = (
                x_t - sqrt_one_minus_alpha_bar * parsed_epsilon_0_to_t
            ) / sqrt_alpha_bar
        else:
            assert x0_pred is not None
            parsed_x0 = x0_pred
            parsed_epsilon_0_to_t = (
                x_t - sqrt_alpha_bar * parsed_x0
            ) / sqrt_one_minus_alpha_bar
        return parsed_x0, parsed_epsilon_0_to_t

    @staticmethod
    def _randnLike(samples: Tensor, generator: Optional[torch.Generator]) -> Tensor:
        return torch.randn(
            samples.shape,
            dtype=samples.dtype,
            device=samples.device,
            generator=generator,
        )

    def diffuseStep(self, x_t: Tensor, t: Timestep, epsilon_t_to_tp1: Tensor) -> Tensor:
        """Apply forward transition ``t`` using the supplied step noise.

        At ``t=0`` the input is clean data and the output is schedule index 0.
        For later transitions, the input is index ``t-1`` and the output is
        index ``t``. Parameter names remain unchanged for compatibility.
        """
        self._validateSample(x_t, "x_t")
        self._validateMatchingTensor(x_t, epsilon_t_to_tp1, "epsilon_t_to_tp1")
        sqrt_alpha = self._extractCoefficient("sqrt_alphas", t, x_t)
        sqrt_one_minus_alpha = self._extractCoefficient("sqrt_one_minus_alphas", t, x_t)
        return sqrt_alpha * x_t + sqrt_one_minus_alpha * epsilon_t_to_tp1

    def diffuse(
        self,
        x_0: Tensor,
        t: Timestep,
        epsilon_0_to_t: Optional[Tensor] = None,
        *,
        generator: Optional[torch.Generator] = None,
        noise: Optional[Tensor] = None,
    ) -> Tensor:
        """Diffuse clean samples using accumulated noise ``epsilon_0_to_t``.

        ``noise`` remains an accepted keyword alias for compatibility.
        """
        self._validateSample(x_0, "x_0")
        if epsilon_0_to_t is not None and noise is not None:
            raise ValueError("provide only epsilon_0_to_t or its legacy alias noise")
        if epsilon_0_to_t is None:
            epsilon_0_to_t = noise
        if epsilon_0_to_t is None:
            epsilon_0_to_t = self._randnLike(x_0, generator)
        else:
            self._validateMatchingTensor(x_0, epsilon_0_to_t, "epsilon_0_to_t")
        sqrt_alpha_bar = self._extractCoefficient("sqrt_alpha_bars", t, x_0)
        sqrt_one_minus_alpha_bar = self._extractCoefficient(
            "sqrt_one_minus_alpha_bars", t, x_0
        )
        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * epsilon_0_to_t

    def computeVelocity(
        self,
        x_0: Tensor,
        epsilon_0_to_t: Optional[Tensor] = None,
        t: Optional[Timestep] = None,
        *,
        epsilon: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute the velocity target for clean samples and accumulated noise."""
        self._validateSample(x_0, "x_0")
        if epsilon_0_to_t is not None and epsilon is not None:
            raise ValueError("provide only epsilon_0_to_t or its legacy alias epsilon")
        if epsilon_0_to_t is None:
            epsilon_0_to_t = epsilon
        if epsilon_0_to_t is None:
            raise TypeError("epsilon_0_to_t is required")
        if t is None:
            raise TypeError("t is required")
        self._validateMatchingTensor(x_0, epsilon_0_to_t, "epsilon_0_to_t")
        sqrt_alpha_bar = self._extractCoefficient("sqrt_alpha_bars", t, x_0)
        sqrt_one_minus_alpha_bar = self._extractCoefficient(
            "sqrt_one_minus_alpha_bars", t, x_0
        )
        return sqrt_alpha_bar * epsilon_0_to_t - sqrt_one_minus_alpha_bar * x_0

    def _extractNoise(self, x_0: Tensor, x_t: Tensor, t: Timestep) -> Tensor:
        self._validateSample(x_0, "x_0")
        self._validateMatchingTensor(x_0, x_t, "x_t")
        sqrt_alpha_bar = self._extractCoefficient("sqrt_alpha_bars", t, x_0)
        sqrt_one_minus_alpha_bar = self._extractCoefficient(
            "sqrt_one_minus_alpha_bars", t, x_0
        )
        epsilon_0_to_t = (x_t - sqrt_alpha_bar * x_0) / sqrt_one_minus_alpha_bar
        return epsilon_0_to_t

    def _extractVelocity(self, x_0: Tensor, x_t: Tensor, t: Timestep) -> Tensor:
        self._validateSample(x_0, "x_0")
        self._validateMatchingTensor(x_0, x_t, "x_t")
        sqrt_alpha_bar = self._extractCoefficient("sqrt_alpha_bars", t, x_0)
        sqrt_one_minus_alpha_bar = self._extractCoefficient(
            "sqrt_one_minus_alpha_bars", t, x_0
        )
        return (sqrt_alpha_bar * x_t - x_0) / sqrt_one_minus_alpha_bar
