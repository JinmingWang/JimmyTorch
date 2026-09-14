"""Denoising Diffusion Probabilistic Model process utilities."""

from __future__ import annotations

import warnings
from typing import Any, Callable, Optional, Tuple, Union

import torch
from tqdm import tqdm

from ._discrete import ScaleMode, Tensor, Timestep, _DiscreteGaussianDiffusion


class DDPM(_DiscreteGaussianDiffusion):
    """Denoising Diffusion Probabilistic Model process and ancestral sampler.

    Schedule index ``0`` is the first noisy state. ``clean_timestep`` (``-1``)
    identifies clean data at the end of the reverse process.

    Reference:
        Ho, Jain, and Abbeel, "Denoising Diffusion Probabilistic Models,"
        2020. https://arxiv.org/abs/2006.11239
    """

    def __init__(
        self,
        min_beta: float = 0.0001,
        max_beta: float = 0.002,
        max_diffusion_step: int = 100,
        device: str = "cuda",
        scale_mode: ScaleMode = "linear",
    ) -> None:
        super().__init__(
            min_beta=min_beta,
            max_beta=max_beta,
            max_diffusion_step=max_diffusion_step,
            device=device,
            scale_mode=scale_mode,
        )

        # Backward-compatible schedule aliases used by older call sites.
        self.beta = self.betas
        self.alpha = self.alphas
        self.αbar = self.alpha_bars
        self.sqrt_αbar = self.sqrt_alpha_bars
        self.sqrt_1_m_αbar = self.sqrt_1m_alpha_bars

    @staticmethod
    def _warnLegacyEndpoint(
        current_timesteps: Tensor, previous_timesteps: Tensor
    ) -> None:
        ambiguous_endpoint = (previous_timesteps == 0) & (current_timesteps > 0)
        if torch.any(ambiguous_endpoint).item():
            warnings.warn(
                "t_prev=0 now denotes the first noisy schedule index. Use "
                "DDPM.clean_timestep (-1) for the clean-data endpoint.",
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
                adjacent_only=True,
            )
        else:
            current_timesteps = torch.as_tensor(t, device=x_t.device, dtype=torch.long)
            previous_timesteps = torch.as_tensor(
                t_prev, device=x_t.device, dtype=torch.long
            )
        if warn_legacy_endpoint:
            self._warnLegacyEndpoint(current_timesteps, previous_timesteps)

        parsed_x0, _ = self._parsePrediction(
            x_t=x_t,
            t=current_timesteps,
            x0_pred=x0_pred,
            epsilon_0_to_t_pred=epsilon_0_to_t_pred,
            v_pred=v_pred,
            timestep_is_normalized=True,
        )
        posterior_x0_coefficient = self._extractNormalizedCoefficient(
            "posterior_mean_x0_coefficients", current_timesteps, x_t
        )
        posterior_xt_coefficient = self._extractNormalizedCoefficient(
            "posterior_mean_xt_coefficients", current_timesteps, x_t
        )
        posterior_mean = (
            posterior_x0_coefficient * parsed_x0 + posterior_xt_coefficient * x_t
        )

        nonzero_mask = self._reshapeCoefficient(
            (current_timesteps != 0).to(dtype=x_t.dtype), x_t
        )
        # Avoid float32 cancellation in the analytically exact final posterior.
        posterior_mean = (
            nonzero_mask * posterior_mean + (1.0 - nonzero_mask) * parsed_x0
        )
        should_sample_noise = (
            torch.any(current_timesteps != 0).item()
            if sample_noise is None
            else sample_noise
        )
        if should_sample_noise:
            posterior_standard_deviation = torch.sqrt(
                self._extractNormalizedCoefficient(
                    "posterior_variance", current_timesteps, x_t
                )
            )
            reverse_step_noise = self._randnLike(x_t, generator)
            previous_samples = (
                posterior_mean
                + nonzero_mask * posterior_standard_deviation * reverse_step_noise
            )
        else:
            previous_samples = posterior_mean

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
        """Sample one adjacent reverse step from ``t`` to ``t_prev``.

        Exactly one prediction must be provided. DDPM only permits adjacent
        transitions: ``t_prev == t - 1``. The final transition is therefore
        ``0 -> clean_timestep``. Use ``epsilon_0_to_t_pred`` for accumulated
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
        """Denoise from index ``T-1`` through index ``0`` to clean data.

        ``pred_func`` is called exactly once for every configured noise level.
        Its timestep argument is a batch-shaped tensor containing the current
        schedule index.
        """
        self._validateSample(x_T, "x_T")
        current_samples = x_T.clone()
        batch_size = current_samples.shape[0]
        sampling_schedule = list(range(self.T - 1, -1, -1))
        sampling_pairs = [
            (current_timestep, current_timestep - 1)
            for current_timestep in sampling_schedule
        ]
        sampling_iterator = (
            tqdm(sampling_pairs, desc="DDPM sampling") if verbose else sampling_pairs
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
                sample_noise=current_timestep > 0,
            )
        return current_samples

    def combineNoise(
        self,
        epsilon_0_to_t: Optional[Tensor] = None,
        epsilon_t_to_tp1: Optional[Tensor] = None,
        t: Optional[Timestep] = None,
        *,
        eps_0_to_t: Optional[Tensor] = None,
        eps_t_to_tp1: Optional[Tensor] = None,
    ) -> Tensor:
        """Derive ``epsilon_0_to_tp1`` from dependent noise ranges.

        ``epsilon_t_to_tp1`` is the single-step noise and ``epsilon_0_to_t``
        is the accumulated noise. The abbreviated ``eps_*`` keywords remain
        accepted for compatibility.
        """
        if epsilon_0_to_t is not None and eps_0_to_t is not None:
            raise ValueError(
                "provide only epsilon_0_to_t or its legacy alias eps_0_to_t"
            )
        if epsilon_t_to_tp1 is not None and eps_t_to_tp1 is not None:
            raise ValueError(
                "provide only epsilon_t_to_tp1 or its legacy alias eps_t_to_tp1"
            )
        if epsilon_0_to_t is None:
            epsilon_0_to_t = eps_0_to_t
        if epsilon_t_to_tp1 is None:
            epsilon_t_to_tp1 = eps_t_to_tp1
        if epsilon_0_to_t is None:
            raise TypeError("epsilon_0_to_t is required")
        if epsilon_t_to_tp1 is None:
            raise TypeError("epsilon_t_to_tp1 is required")
        if t is None:
            raise TypeError("t is required")

        self._validateSample(epsilon_0_to_t, "epsilon_0_to_t")
        self._validateMatchingTensor(
            epsilon_0_to_t, epsilon_t_to_tp1, "epsilon_t_to_tp1"
        )
        current_timesteps = self._normalizeTimestep(
            t, epsilon_0_to_t.shape[0], "t", allow_clean=False
        )
        sqrt_alpha = self._extractCoefficient(
            "sqrt_alphas", current_timesteps, epsilon_0_to_t
        )
        sqrt_one_minus_alpha = self._extractCoefficient(
            "sqrt_one_minus_alphas", current_timesteps, epsilon_0_to_t
        )
        previous_alpha_bar = self._extractCoefficient(
            "alpha_bars", current_timesteps - 1, epsilon_0_to_t, allow_clean=True
        )
        sqrt_one_minus_previous_alpha_bar = torch.sqrt(
            (1.0 - previous_alpha_bar).clamp_min(0.0)
        )
        sqrt_one_minus_current_alpha_bar = self._extractCoefficient(
            "sqrt_one_minus_alpha_bars", current_timesteps, epsilon_0_to_t
        )
        epsilon_0_to_tp1 = (
            sqrt_alpha * sqrt_one_minus_previous_alpha_bar * epsilon_0_to_t
            + sqrt_one_minus_alpha * epsilon_t_to_tp1
        ) / sqrt_one_minus_current_alpha_bar
        first_step_mask = self._reshapeCoefficient(
            (current_timesteps == 0).to(dtype=epsilon_0_to_t.dtype),
            epsilon_0_to_t,
        )
        return (
            first_step_mask * epsilon_t_to_tp1
            + (1.0 - first_step_mask) * epsilon_0_to_tp1
        )
