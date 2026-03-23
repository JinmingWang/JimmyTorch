# encoding: utf-8

import torch
from typing import Callable, Literal, Tuple, Optional, Union
from tqdm import tqdm
from math import log

Tensor = torch.Tensor

class DDIM:
    """
    Denoising Diffusion Implicit Models (DDIM) implementation.

    DDIM provides a deterministic sampling process that can skip steps,
    allowing for faster generation compared to DDPM.
    """

    def __init__(self,
                 min_beta: float = 0.0001,
                 max_beta: float = 0.002,
                 max_diffusion_step: int = 100,
                 device: str = 'cuda',
                 scale_mode: Literal["linear", "quadratic", "log"] = "linear",
                 skip_step: int = 1):
        """
        Initializes the DDIM model with the given parameters.

        :param min_beta: Minimum beta value for the diffusion process.
        :param max_beta: Maximum beta value for the diffusion process.
        :param max_diffusion_step: Total number of diffusion steps.
        :param device: Device to perform computations on ('cuda' or 'cpu').
        :param scale_mode: Mode for scaling beta values ('linear', 'quadratic', or 'log').
        :param skip_step: Number of steps to skip during denoising (1 means no skipping).
        """
        # Compute beta schedule
        if scale_mode == "quadratic":
            betas = torch.linspace(min_beta ** 0.5, max_beta ** 0.5, max_diffusion_step, device=device) ** 2
        elif scale_mode == "log":
            betas = torch.exp(torch.linspace(log(min_beta), log(max_beta), max_diffusion_step, device=device))
        else:
            betas = torch.linspace(min_beta, max_beta, max_diffusion_step, device=device)

        self.skip_step = skip_step
        self.device = device
        self.T = max_diffusion_step

        # Compute alphas and cumulative products
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        # Store precomputed values (T, 1) for broadcasting
        self.betas = betas.view(-1, 1)
        self.alphas = alphas.view(-1, 1)
        self.alpha_bars = alpha_bars.view(-1, 1)
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars).view(-1, 1)
        self.sqrt_1m_alpha_bars = torch.sqrt(1.0 - alpha_bars).view(-1, 1)


    def diffuseStep(self, x_t: Tensor, t: Union[int, Tensor], epsilon_t_to_tp1: Tensor) -> Tensor:
        """
        Diffuse one step from x_t to x_{t+1} using the noise epsilon.

        Formula: x_{t+1} = sqrt(alpha[t]) * x_t + sqrt(1 - alpha[t]) * epsilon

        :param x_t: Sample at timestep t, shape (B, ...)
        :param t: Timestep, either int or Tensor of shape (B,)
        :param epsilon_t_to_tp1: Noise to add, same shape as x_t
        :return: Noisy sample at timestep t+1, same shape as x_t
        """
        original_shape = x_t.shape
        x_t_flat = x_t.flatten(1)
        epsilon_flat = epsilon_t_to_tp1.flatten(1)

        sqrt_alpha_t = torch.sqrt(self.alphas[t])
        sqrt_1m_alpha_t = torch.sqrt(1 - self.alphas[t])

        x_tp1_flat = sqrt_alpha_t * x_t_flat + sqrt_1m_alpha_t * epsilon_flat
        return x_tp1_flat.view(original_shape)


    def diffuse(self, x_0: Tensor, t: Union[int, Tensor], noise: Optional[Tensor] = None) -> Tensor:
        """
        Forward diffusion: Add noise to x_0 to get x_t.

        Formula: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise

        :param x_0: Original sample at timestep 0, shape (B, ...)
        :param t: Timestep(s), either int or Tensor of shape (B,)
        :param noise: Gaussian noise, same shape as x_0. If None, will be sampled.
        :return: Noisy sample x_t, same shape as x_0
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        original_shape = x_0.shape
        x_0_flat = x_0.flatten(1)
        noise_flat = noise.flatten(1)

        # Handle both int and tensor timesteps
        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t]
        sqrt_1m_alpha_bar_t = self.sqrt_1m_alpha_bars[t]

        x_t_flat = sqrt_alpha_bar_t * x_0_flat + sqrt_1m_alpha_bar_t * noise_flat
        return x_t_flat.view(original_shape)


    def denoiseStep(self,
                    x_t: Tensor,
                    t: Union[int, Tensor],
                    t_prev: Union[int, Tensor],
                    x0_pred: Optional[Tensor] = None,
                    epsilon_pred: Optional[Tensor] = None,
                    v_pred: Optional[Tensor] = None,
                    need_x0: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Single DDIM denoising step from timestep t to t_prev.

        Exactly one of x0_pred, epsilon_pred, or v_pred must be provided.

        :param x_t: Current noisy sample at timestep t, shape (B, ...)
        :param t: Current timestep, either int or Tensor of shape (B,)
        :param t_prev: Previous (smaller) timestep to denoise to
        :param x0_pred: Predicted x_0, if predicting x0 directly
        :param epsilon_pred: Predicted noise, if predicting noise
        :param v_pred: Predicted velocity, if predicting velocity
        :param need_x0: If True, also return the predicted x_0
        :return: x_{t_prev} or (x_{t_prev}, x_0_pred) if need_x0=True
        """
        original_shape = x_t.shape
        x_t_flat = x_t.flatten(1)

        # Get precomputed values for timestep t
        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t]
        sqrt_1m_alpha_bar_t = self.sqrt_1m_alpha_bars[t]

        # Convert prediction to x0 and epsilon
        if v_pred is not None:
            # v-prediction: v = sqrt(alpha_bar_t) * epsilon - sqrt(1 - alpha_bar_t) * x_0
            # Solve for x_0 and epsilon:
            # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
            # v = sqrt(alpha_bar_t) * epsilon - sqrt(1 - alpha_bar_t) * x_0
            v_flat = v_pred.flatten(1)
            x0_pred_flat = sqrt_alpha_bar_t * x_t_flat - sqrt_1m_alpha_bar_t * v_flat
            epsilon_pred_flat = sqrt_1m_alpha_bar_t * x_t_flat + sqrt_alpha_bar_t * v_flat
        elif epsilon_pred is not None:
            # Noise prediction: solve for x_0 from x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
            epsilon_pred_flat = epsilon_pred.flatten(1)
            x0_pred_flat = (x_t_flat - sqrt_1m_alpha_bar_t * epsilon_pred_flat) / sqrt_alpha_bar_t
        elif x0_pred is not None:
            # Direct x_0 prediction: solve for epsilon
            x0_pred_flat = x0_pred.flatten(1)
            epsilon_pred_flat = (x_t_flat - sqrt_alpha_bar_t * x0_pred_flat) / sqrt_1m_alpha_bar_t
        else:
            raise ValueError("Must provide exactly one of x0_pred, epsilon_pred, or v_pred")

        # DDIM deterministic sampling
        # If t_prev == 0, return x_0 directly
        # Otherwise, compute x_{t_prev} = sqrt(alpha_bar_{t_prev}) * x_0 + sqrt(1 - alpha_bar_{t_prev}) * epsilon

        # Create mask for final step (when we should return x_0 directly)
        if isinstance(t_prev, int):
            is_final = (t_prev == 0)
        else:
            is_final = (t_prev == 0).to(x0_pred_flat.dtype).view(-1, 1)

        if isinstance(is_final, bool):
            if is_final:
                x_prev_flat = x0_pred_flat
            else:
                sqrt_alpha_bar_prev = self.sqrt_alpha_bars[t_prev]
                sqrt_1m_alpha_bar_prev = self.sqrt_1m_alpha_bars[t_prev]
                x_prev_flat = sqrt_alpha_bar_prev * x0_pred_flat + sqrt_1m_alpha_bar_prev * epsilon_pred_flat
        else:
            sqrt_alpha_bar_prev = self.sqrt_alpha_bars[t_prev]
            sqrt_1m_alpha_bar_prev = self.sqrt_1m_alpha_bars[t_prev]
            x_prev_noised = sqrt_alpha_bar_prev * x0_pred_flat + sqrt_1m_alpha_bar_prev * epsilon_pred_flat
            x_prev_flat = is_final * x0_pred_flat + (1 - is_final) * x_prev_noised

        x_prev = x_prev_flat.view(original_shape)

        if need_x0:
            return x_prev, x0_pred_flat.view(original_shape)
        return x_prev


    @torch.no_grad()
    def denoise(self,
                x_T: Tensor,
                pred_func: Callable[[Tensor, Tensor], Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]],
                verbose: bool = False,
                **pred_func_args) -> Tensor:
        """
        Denoises a sample from the final timestep T to timestep 0.

        :param x_T: Noisy sample at timestep T, shape (B, ...)
        :param pred_func: Prediction function that takes (x_t, t) and returns (x0_pred, epsilon_pred, v_pred).
                         Should return exactly one non-None prediction.
        :param verbose: Whether to display a progress bar
        :param pred_func_args: Additional arguments to pass to pred_func
        :return: Denoised sample at timestep 0
        """
        x_t = x_T.clone()
        batch_size = x_T.shape[0]

        # Create timestep schedule with skip_step
        timesteps = list(range(self.T - 1, -1, -self.skip_step))
        if timesteps[-1] != 0:
            timesteps.append(0)

        iterator = tqdm(timesteps, desc="DDIM sampling") if verbose else timesteps

        for i, t in enumerate(iterator):
            # Create timestep tensor for batch
            t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=self.device)

            # Get prediction from model
            x0_pred, epsilon_pred, v_pred = pred_func(x_t, t_tensor, **pred_func_args)

            # Determine next timestep
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            t_prev_tensor = torch.full((batch_size,), t_prev, dtype=torch.long, device=self.device)

            # Denoise one step
            x_t = self.denoiseStep(x_t, t_tensor, t_prev_tensor, x0_pred, epsilon_pred, v_pred)

        return x_t


    def extractNoise(self, x_0: Tensor, x_t: Tensor, t: Union[int, Tensor]) -> Tensor:
        """
        Extract noise from x_t given x_0.

        According to diffusion formula:
        x_t = sqrt(alpha_bar[t]) * x_0 + sqrt(1 - alpha_bar[t]) * epsilon

        Solve for epsilon:
        epsilon = (x_t - sqrt(alpha_bar[t]) * x_0) / sqrt(1 - alpha_bar[t])

        :param x_0: Original sample at timestep 0
        :param x_t: Noisy sample at timestep t
        :param t: Timestep, either int or Tensor
        :return: Extracted noise epsilon
        """
        original_shape = x_0.shape
        x_0_flat = x_0.flatten(1)
        x_t_flat = x_t.flatten(1)

        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t]
        sqrt_1m_alpha_bar_t = self.sqrt_1m_alpha_bars[t]

        epsilon = (x_t_flat - sqrt_alpha_bar_t * x_0_flat) / sqrt_1m_alpha_bar_t
        return epsilon.view(original_shape)


    def computeVelocity(self, x_0: Tensor, epsilon: Tensor, t: Union[int, Tensor]) -> Tensor:
        """
        Compute velocity prediction target from x_0 and noise.

        v = sqrt(alpha_bar[t]) * epsilon - sqrt(1 - alpha_bar[t]) * x_0

        :param x_0: Original sample at timestep 0
        :param epsilon: Noise
        :param t: Timestep, either int or Tensor
        :return: Velocity v
        """
        original_shape = x_0.shape
        x_0_flat = x_0.flatten(1)
        epsilon_flat = epsilon.flatten(1)

        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t]
        sqrt_1m_alpha_bar_t = self.sqrt_1m_alpha_bars[t]

        v = sqrt_alpha_bar_t * epsilon_flat - sqrt_1m_alpha_bar_t * x_0_flat
        return v.view(original_shape)


    def extractVelocity(self, x_0: Tensor, x_t: Tensor, t: Union[int, Tensor]) -> Tensor:
        """
        Extract velocity from x_t given x_0.

        According to diffusion formula:
        x_t = sqrt(alpha_bar[t]) * x_0 + sqrt(1 - alpha_bar[t]) * epsilon

        And velocity formula:
        v = sqrt(alpha_bar[t]) * epsilon - sqrt(1 - alpha_bar[t]) * x_0

        Combining these:
        v = (sqrt(alpha_bar[t]) * x_t - x_0) / sqrt(1 - alpha_bar[t])

        :param x_0: Original sample at timestep 0
        :param x_t: Noisy sample at timestep t
        :param t: Timestep, either int or Tensor
        :return: Extracted velocity v
        """
        original_shape = x_0.shape
        x_0_flat = x_0.flatten(1)
        x_t_flat = x_t.flatten(1)

        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t]
        sqrt_1m_alpha_bar_t = self.sqrt_1m_alpha_bars[t]

        v = (sqrt_alpha_bar_t * x_t_flat - x_0_flat) / sqrt_1m_alpha_bar_t
        return v.view(original_shape)


