"""Denoising Diffusion Implicit Model process utilities."""

from __future__ import annotations

import warnings
from numbers import Integral, Real
from typing import Any, Callable, Optional, Tuple, Union

import torch
from tqdm import tqdm

from ._discrete import ScaleMode, Tensor, Timestep, _DiscreteGaussianDiffusion


class DDIM(_DiscreteGaussianDiffusion):
    """Denoising Diffusion Implicit Model process and sampler.

    ``eta=0`` gives deterministic DDIM sampling. Positive ``eta`` values add
    controlled stochasticity. Schedule index ``0`` is the first noisy state;
    ``clean_timestep`` (``-1``) identifies clean data.

    Reference:
        Song, Meng, and Ermon, "Denoising Diffusion Implicit Models," 2020.
        https://arxiv.org/abs/2010.02502
    """

    def __init__(
        self,
        min_beta: float = 0.0001,
        max_beta: float = 0.002,
        max_diffusion_step: int = 100,
        device: str = "cuda",
        scale_mode: ScaleMode = "linear",
        skip_step: int = 1,
        eta: float = 0.0,
    ) -> None:
        if not isinstance(skip_step, Integral) or isinstance(skip_step, bool):
            raise TypeError("skip_step must be an integer")
        if skip_step < 1:
            raise ValueError("skip_step must be at least 1")
        if not isinstance(eta, Real) or isinstance(eta, bool):
            raise TypeError("eta must be a real number")
        if not 0 <= eta <= 1:
            raise ValueError("eta must be in the range [0, 1]")

        super().__init__(
            min_beta=min_beta,
            max_beta=max_beta,
            max_diffusion_step=max_diffusion_step,
            device=device,
            scale_mode=scale_mode,
        )
        self.skip_step = int(skip_step)
        self.eta = float(eta)

    @staticmethod
    def _warnLegacyEndpoint(
        current_timesteps: Tensor, previous_timesteps: Tensor
    ) -> None:
        ambiguous_endpoint = (previous_timesteps == 0) & (current_timesteps > 0)
        if torch.any(ambiguous_endpoint).item():
            warnings.warn(
                "t_prev=0 now denotes the first noisy schedule index. Use "
                "DDIM.clean_timestep (-1) for the clean-data endpoint.",
                DeprecationWarning,
                stacklevel=3,
            )

    def _denoiseStepImpl(
        self,
        x_t: Tensor,
        t: Timestep,
        t_prev: Timestep,
        x0_pred: Optional[Tensor],
        epsilon_0_to_t_pred: Optional[Tensor],
        v_pred: Optional[Tensor],
        need_x0: bool,
        generator: Optional[torch.Generator],
        warn_legacy_endpoint: bool,
        validate_transition: bool,
        sample_noise: Optional[bool] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        self._validateSample(x_t, "x_t")
        if validate_transition:
            current_timesteps, previous_timesteps = self._validateTransition(
                current_timestep=t,
                previous_timestep=t_prev,
                batch_size=x_t.shape[0],
                adjacent_only=False,
            )
        else:
            current_timesteps = torch.as_tensor(t, device=x_t.device, dtype=torch.long)
            previous_timesteps = torch.as_tensor(
                t_prev, device=x_t.device, dtype=torch.long
            )
        if warn_legacy_endpoint:
            self._warnLegacyEndpoint(current_timesteps, previous_timesteps)

        parsed_x0, parsed_epsilon = self._parsePrediction(
            x_t=x_t,
            t=current_timesteps,
            x0_pred=x0_pred,
            epsilon_0_to_t_pred=epsilon_0_to_t_pred,
            v_pred=v_pred,
            timestep_is_normalized=True,
        )
        previous_alpha_bar = self._extractNormalizedCoefficient(
            "alpha_bars", previous_timesteps, x_t, allow_clean=True
        )
        if self.eta == 0:
            previous_samples = (
                torch.sqrt(previous_alpha_bar) * parsed_x0
                + torch.sqrt((1.0 - previous_alpha_bar).clamp_min(0.0)) * parsed_epsilon
            )
        else:
            current_alpha_bar = self._extractNormalizedCoefficient(
                "alpha_bars", current_timesteps, x_t
            )
            variance_ratio = (
                (1.0 - previous_alpha_bar)
                / (1.0 - current_alpha_bar)
                * (1.0 - current_alpha_bar / previous_alpha_bar)
            ).clamp_min(0.0)
            noise_standard_deviation = self.eta * torch.sqrt(variance_ratio)
            epsilon_coefficient = torch.sqrt(
                (
                    1.0 - previous_alpha_bar - noise_standard_deviation.square()
                ).clamp_min(0.0)
            )
            previous_samples = (
                torch.sqrt(previous_alpha_bar) * parsed_x0
                + epsilon_coefficient * parsed_epsilon
            )
            should_sample_noise = (
                torch.any(noise_standard_deviation > 0).item()
                if sample_noise is None
                else sample_noise
            )
            if should_sample_noise:
                ddim_reverse_step_noise = self._randnLike(x_t, generator)
                previous_samples = (
                    previous_samples
                    + noise_standard_deviation * ddim_reverse_step_noise
                )

        if need_x0:
            return previous_samples, parsed_x0
        return previous_samples

    def denoiseStep(
        self,
        x_t: Tensor,
        t: Timestep,
        t_prev: Timestep,
        x0_pred: Optional[Tensor] = None,
        epsilon_pred: Optional[Tensor] = None,
        v_pred: Optional[Tensor] = None,
        need_x0: bool = False,
        *,
        epsilon_0_to_t_pred: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Sample one strictly descending DDIM transition.

        ``t_prev`` may skip noise levels and is ``clean_timestep`` for the
        final transition. Use ``epsilon_0_to_t_pred`` for accumulated
        predicted noise; ``epsilon_pred`` remains its compatibility alias.
        """
        if epsilon_pred is not None and epsilon_0_to_t_pred is not None:
            raise ValueError(
                "provide only epsilon_0_to_t_pred or its legacy alias epsilon_pred"
            )
        accumulated_epsilon_prediction = (
            epsilon_0_to_t_pred if epsilon_0_to_t_pred is not None else epsilon_pred
        )
        return self._denoiseStepImpl(
            x_t=x_t,
            t=t,
            t_prev=t_prev,
            x0_pred=x0_pred,
            epsilon_0_to_t_pred=accumulated_epsilon_prediction,
            v_pred=v_pred,
            need_x0=need_x0,
            generator=generator,
            warn_legacy_endpoint=True,
            validate_transition=True,
        )

    @torch.no_grad()
    def denoise(
        self,
        x_T: Tensor,
        pred_func: Callable[
            ...,
            Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]],
        ],
        verbose: bool = False,
        *,
        generator: Optional[torch.Generator] = None,
        **pred_func_args: Any,
    ) -> Tensor:
        """Denoise over the configured skipped schedule to clean data.

        ``pred_func`` receives a batch-shaped tensor containing the current
        schedule index and is called once for every selected noise level.
        """
        self._validateSample(x_T, "x_T")
        current_samples = x_T.clone()
        batch_size = current_samples.shape[0]
        sampling_schedule = list(range(self.T - 1, -1, -self.skip_step))
        if sampling_schedule[-1] != 0:
            sampling_schedule.append(0)
        sampling_pairs = list(
            zip(
                sampling_schedule,
                sampling_schedule[1:] + [self.clean_timestep],
            )
        )
        sampling_iterator = (
            tqdm(sampling_pairs, desc="DDIM sampling") if verbose else sampling_pairs
        )

        for current_timestep, previous_timestep in sampling_iterator:
            batch_timesteps = torch.full(
                (batch_size,),
                current_timestep,
                dtype=torch.long,
                device=current_samples.device,
            )
            prediction_output = pred_func(
                current_samples, batch_timesteps, **pred_func_args
            )
            try:
                (
                    x0_prediction,
                    epsilon_0_to_t_prediction,
                    velocity_prediction,
                ) = prediction_output
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "pred_func must return " "(x0_pred, epsilon_0_to_t_pred, v_pred)"
                ) from error
            current_samples = self._denoiseStepImpl(
                x_t=current_samples,
                t=current_timestep,
                t_prev=previous_timestep,
                x0_pred=x0_prediction,
                epsilon_0_to_t_pred=epsilon_0_to_t_prediction,
                v_pred=velocity_prediction,
                need_x0=False,
                generator=generator,
                warn_legacy_endpoint=False,
                validate_transition=False,
                sample_noise=self.eta > 0 and previous_timestep != self.clean_timestep,
            )
        return current_samples

    def extractNoise(self, x_0: Tensor, x_t: Tensor, t: Timestep) -> Tensor:
        """Recover accumulated ``epsilon_0_to_t`` from ``x_0`` and ``x_t``."""
        return self._extractNoise(x_0=x_0, x_t=x_t, t=t)

    def extractVelocity(self, x_0: Tensor, x_t: Tensor, t: Timestep) -> Tensor:
        """Recover the velocity target from ``x_0`` and ``x_t``."""
        return self._extractVelocity(x_0=x_0, x_t=x_t, t=t)
