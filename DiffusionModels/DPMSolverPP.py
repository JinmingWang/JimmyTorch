"""DPM-Solver++ sampling for discrete Gaussian diffusion schedules."""

from __future__ import annotations

from numbers import Integral
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import torch
from tqdm import tqdm

from ._discrete import ScaleMode, Tensor, Timestep, _DiscreteGaussianDiffusion


class DPMSolverPP(_DiscreteGaussianDiffusion):
    """First- or second-order multistep DPM-Solver++ sampler.

    This data-prediction solver accepts the same three-item callback contract
    as :class:`DDPM` and :class:`DDIM`: exactly one of ``x0_pred``,
    ``epsilon_0_to_t_pred``, or ``v_pred`` must be populated. Schedule index
    ``0`` is the final noisy level and ``clean_timestep`` (``-1``) is clean.

    Reference:
        Lu et al., "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion
        Probabilistic Models," 2022. https://arxiv.org/abs/2211.01095
    """

    def __init__(
        self,
        min_beta: float = 0.0001,
        max_beta: float = 0.002,
        max_diffusion_step: int = 100,
        device: str = "cuda",
        scale_mode: ScaleMode = "linear",
        num_inference_steps: Optional[int] = None,
        solver_order: int = 2,
    ) -> None:
        super().__init__(
            min_beta=min_beta,
            max_beta=max_beta,
            max_diffusion_step=max_diffusion_step,
            device=device,
            scale_mode=scale_mode,
        )
        if num_inference_steps is None:
            num_inference_steps = min(20, self.T)
        if not isinstance(num_inference_steps, Integral) or isinstance(
            num_inference_steps, bool
        ):
            raise TypeError("num_inference_steps must be an integer")
        minimum_steps = 1 if self.T == 1 else 2
        if not minimum_steps <= num_inference_steps <= self.T:
            raise ValueError(
                f"num_inference_steps must be in [{minimum_steps}, {self.T}]"
            )
        if not isinstance(solver_order, Integral) or isinstance(solver_order, bool):
            raise TypeError("solver_order must be an integer")
        if solver_order not in {1, 2}:
            raise ValueError("solver_order must be 1 or 2")

        self.num_inference_steps = int(num_inference_steps)
        self.solver_order = int(solver_order)
        marginal_lambdas = torch.log(
            self._base_schedules["sqrt_alpha_bars"]
        ) - torch.log(self._base_schedules["sqrt_one_minus_alpha_bars"])
        self._base_schedules["marginal_lambdas"] = marginal_lambdas

    def samplingTimesteps(
        self, timesteps: Optional[Sequence[int]] = None
    ) -> Tuple[int, ...]:
        """Return a validated descending inference schedule ending at index ``0``."""
        if timesteps is None:
            generated_timesteps = (
                torch.linspace(
                    self.T - 1,
                    0,
                    self.num_inference_steps,
                    dtype=torch.float64,
                    device=self._torch_device,
                )
                .round()
                .to(dtype=torch.long)
            )
            sampling_schedule = tuple(int(value) for value in generated_timesteps)
        else:
            if isinstance(timesteps, Tensor):
                if timesteps.ndim != 1:
                    raise ValueError("timesteps must be one-dimensional")
                if timesteps.dtype not in {
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                    torch.uint8,
                }:
                    raise TypeError("timesteps must use an integer dtype")
                sampling_schedule = tuple(int(value) for value in timesteps.cpu())
            else:
                if not isinstance(timesteps, Sequence) or isinstance(
                    timesteps, (str, bytes)
                ):
                    raise TypeError("timesteps must be a sequence of integers")
                sampling_schedule = tuple(timesteps)
            if any(
                not isinstance(value, Integral) or isinstance(value, bool)
                for value in sampling_schedule
            ):
                raise TypeError("timesteps must contain only integers")
            sampling_schedule = tuple(int(value) for value in sampling_schedule)

        if not sampling_schedule:
            raise ValueError("timesteps must not be empty")
        if sampling_schedule[0] != self.T - 1:
            raise ValueError(f"timesteps must start at schedule index {self.T - 1}")
        if sampling_schedule[-1] != 0:
            raise ValueError("timesteps must end at schedule index 0")
        if any(value < 0 or value >= self.T for value in sampling_schedule):
            raise ValueError(f"timesteps must be in [0, {self.T - 1}]")
        if any(
            previous_timestep <= current_timestep
            for previous_timestep, current_timestep in zip(
                sampling_schedule, sampling_schedule[1:]
            )
        ):
            raise ValueError("timesteps must be strictly descending")
        return sampling_schedule

    def _firstOrderUpdate(
        self,
        x_t: Tensor,
        current_timesteps: Tensor,
        previous_timesteps: Tensor,
        x0_prediction: Tensor,
    ) -> Tensor:
        clean_mask = previous_timesteps == self.clean_timestep
        noisy_previous_timesteps = previous_timesteps.clamp_min(0)
        current_sigma = self._extractNormalizedCoefficient(
            "sqrt_one_minus_alpha_bars", current_timesteps, x_t
        )
        previous_sigma = self._extractNormalizedCoefficient(
            "sqrt_one_minus_alpha_bars", noisy_previous_timesteps, x_t
        )
        previous_alpha = self._extractNormalizedCoefficient(
            "sqrt_alpha_bars", noisy_previous_timesteps, x_t
        )
        current_lambda = self._extractNormalizedCoefficient(
            "marginal_lambdas", current_timesteps, x_t
        )
        previous_lambda = self._extractNormalizedCoefficient(
            "marginal_lambdas", noisy_previous_timesteps, x_t
        )
        lambda_step = previous_lambda - current_lambda
        first_order_samples = (
            previous_sigma / current_sigma * x_t
            - previous_alpha * torch.expm1(-lambda_step) * x0_prediction
        )
        broadcast_clean_mask = self._reshapeCoefficient(
            clean_mask.to(dtype=x_t.dtype), x_t
        )
        return (
            broadcast_clean_mask * x0_prediction
            + (1.0 - broadcast_clean_mask) * first_order_samples
        )

    def _secondOrderUpdate(
        self,
        x_t: Tensor,
        older_timestep: int,
        current_timestep: int,
        previous_timestep: int,
        older_x0_prediction: Tensor,
        current_x0_prediction: Tensor,
    ) -> Tensor:
        current_tensor = torch.tensor(
            current_timestep, device=x_t.device, dtype=torch.long
        )
        previous_tensor = torch.tensor(
            previous_timestep, device=x_t.device, dtype=torch.long
        )
        older_tensor = torch.tensor(older_timestep, device=x_t.device, dtype=torch.long)
        current_sigma = self._extractNormalizedCoefficient(
            "sqrt_one_minus_alpha_bars", current_tensor, x_t
        )
        previous_sigma = self._extractNormalizedCoefficient(
            "sqrt_one_minus_alpha_bars", previous_tensor, x_t
        )
        previous_alpha = self._extractNormalizedCoefficient(
            "sqrt_alpha_bars", previous_tensor, x_t
        )
        older_lambda = self._extractNormalizedCoefficient(
            "marginal_lambdas", older_tensor, x_t
        )
        current_lambda = self._extractNormalizedCoefficient(
            "marginal_lambdas", current_tensor, x_t
        )
        previous_lambda = self._extractNormalizedCoefficient(
            "marginal_lambdas", previous_tensor, x_t
        )
        previous_lambda_step = current_lambda - older_lambda
        current_lambda_step = previous_lambda - current_lambda
        step_ratio = previous_lambda_step / current_lambda_step
        first_model_difference = (
            current_x0_prediction - older_x0_prediction
        ) / step_ratio
        first_order_coefficient = torch.expm1(-current_lambda_step)
        return (
            previous_sigma / current_sigma * x_t
            - previous_alpha * first_order_coefficient * current_x0_prediction
            - 0.5 * previous_alpha * first_order_coefficient * first_model_difference
        )

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
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Run one stateless first-order DPM-Solver++ transition.

        Second-order updates require prediction history and are therefore
        applied by :meth:`denoise`. The direct method is equivalent to the
        first-order DPM-Solver++/DDIM update.
        """
        self._validateSample(x_t, "x_t")
        if epsilon_pred is not None and epsilon_0_to_t_pred is not None:
            raise ValueError(
                "provide only epsilon_0_to_t_pred or its legacy alias epsilon_pred"
            )
        accumulated_epsilon_prediction = (
            epsilon_0_to_t_pred if epsilon_0_to_t_pred is not None else epsilon_pred
        )
        current_timesteps, previous_timesteps = self._validateTransition(
            current_timestep=t,
            previous_timestep=t_prev,
            batch_size=x_t.shape[0],
            adjacent_only=False,
        )
        parsed_x0, _ = self._parsePrediction(
            x_t=x_t,
            t=current_timesteps,
            x0_pred=x0_pred,
            epsilon_0_to_t_pred=accumulated_epsilon_prediction,
            v_pred=v_pred,
            timestep_is_normalized=True,
        )
        previous_samples = self._firstOrderUpdate(
            x_t,
            current_timesteps,
            previous_timesteps,
            parsed_x0,
        )
        if need_x0:
            return previous_samples, parsed_x0
        return previous_samples

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
        timesteps: Optional[Sequence[int]] = None,
        **pred_func_args: Any,
    ) -> Tensor:
        """Denoise with one model evaluation per selected noisy level.

        ``timesteps`` may override the generated schedule but must start at
        ``T-1``, be strictly descending, and end at ``0``. Order two uses the
        previous data prediction as multistep history and uses a stable final
        data-prediction step from index ``0`` to clean data.
        """
        self._validateSample(x_T, "x_T")
        current_samples = x_T.clone()
        batch_size = current_samples.shape[0]
        sampling_schedule = self.samplingTimesteps(timesteps)
        sampling_pairs = list(
            zip(
                sampling_schedule,
                sampling_schedule[1:] + (self.clean_timestep,),
            )
        )
        sampling_iterator = (
            tqdm(sampling_pairs, desc="DPM-Solver++ sampling")
            if verbose
            else sampling_pairs
        )

        older_x0_prediction: Optional[Tensor] = None
        older_timestep: Optional[int] = None
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
            parsed_x0, _ = self._parsePrediction(
                x_t=current_samples,
                t=batch_timesteps,
                x0_pred=x0_prediction,
                epsilon_0_to_t_pred=epsilon_0_to_t_prediction,
                v_pred=velocity_prediction,
                timestep_is_normalized=True,
            )

            if previous_timestep == self.clean_timestep:
                next_samples = parsed_x0
            elif (
                self.solver_order == 2
                and older_x0_prediction is not None
                and older_timestep is not None
            ):
                next_samples = self._secondOrderUpdate(
                    x_t=current_samples,
                    older_timestep=older_timestep,
                    current_timestep=current_timestep,
                    previous_timestep=previous_timestep,
                    older_x0_prediction=older_x0_prediction,
                    current_x0_prediction=parsed_x0,
                )
            else:
                current_tensor = torch.tensor(
                    current_timestep, device=current_samples.device, dtype=torch.long
                )
                previous_tensor = torch.tensor(
                    previous_timestep,
                    device=current_samples.device,
                    dtype=torch.long,
                )
                next_samples = self._firstOrderUpdate(
                    current_samples,
                    current_tensor,
                    previous_tensor,
                    parsed_x0,
                )

            older_x0_prediction = parsed_x0
            older_timestep = current_timestep
            current_samples = next_samples
        return current_samples
