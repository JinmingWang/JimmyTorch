# encoding: utf-8

import torch
from typing import *
from tqdm import tqdm
from math import log

Tensor = torch.Tensor

class DDIM:

    def __init__(self,
                 min_beta: float = 0.0001,
                 max_beta: float = 0.002,
                 max_diffusion_step: int = 100,
                 device: str = 'cuda',
                 scale_mode: Literal["linear", "quadratic", "log"] = "linear",
                 skip_step=1):
        """
        Initializes the DDIM model with the given parameters.

        :param sample_shape: Shape of the samples to be generated.
        :param min_beta: Minimum beta value for the diffusion process.
        :param max_beta: Maximum beta value for the diffusion process.
        :param max_diffusion_step: Total number of diffusion steps.
        :param device: Device to perform computations on ('cuda' or 'cpu').
        :param scale_mode: Mode for scaling beta values.
        :param skip_step: Number of steps to skip during denoising.
        """
        if scale_mode == "quadratic":
            betas = torch.linspace(min_beta ** 0.5, max_beta ** 0.5, max_diffusion_step).to(device) ** 2
        elif scale_mode == "log":
            betas = torch.exp(torch.linspace(log(min_beta), log(max_beta), max_diffusion_step).to(device))
        else:
            betas = torch.linspace(min_beta, max_beta, max_diffusion_step).to(device)

        self.skip_step = skip_step
        self.device = device

        alphas = 1 - betas
        alpha_bars = torch.empty_like(alphas)
        product = 1
        for i, alpha in enumerate(alphas):
            product *= alpha
            alpha_bars[i] = product
        self.T = max_diffusion_step

        # Shapes: (T,)
        self.beta = betas.view(-1, 1)
        self.alpha = alphas.view(-1, 1)
        self.αbar = alpha_bars.view(-1, 1)
        self.sqrt_αbar = torch.sqrt(alpha_bars).view(-1, 1)
        self.sqrt_1_m_αbar = torch.sqrt(1 - alpha_bars).view(-1, 1)


    def diffuseStep(self, x_t: Tensor, t: int, epsilon_t_to_tp1: Tensor) -> Tensor:
        return torch.sqrt(self.alpha[t]) * x_t + torch.sqrt(1 - self.alpha[t]) * epsilon_t_to_tp1


    def diffuse(self, x_0: Tensor, t: int, epsilon: Tensor) -> Tensor:
        original_shape = x_0.shape
        x_t = self.sqrt_αbar[t] * x_0.flatten(1) + self.sqrt_1_m_αbar[t] * epsilon.flatten(1)
        return x_t.view(*original_shape)


    def denoiseStep(self,
                    x0_pred: Tensor,
                    epsilon_pred: Tensor,
                    v_pred: Tensor,
                    x_tp1: Tensor,
                    t: Tensor,
                    next_t: Tensor,
                    need_x0: bool = False) -> Tensor:
        original_shape = x_tp1.shape

        if v_pred is not None:
            # Flatten for computation
            v_pred = v_pred.flatten(1)
            sqrt_ab = self.sqrt_αbar[t]  # scalar
            sqrt_1mab = self.sqrt_1_m_αbar[t]  # scalar

            x0_pred = sqrt_ab * x_tp1.flatten(1) - sqrt_1mab * v_pred
            epsilon_pred = ((v_pred + sqrt_1mab * x0_pred) / sqrt_ab).flatten(1)
        elif epsilon_pred is not None:
            x0_pred = (x_tp1.flatten(1) - self.sqrt_1_m_αbar[t] * epsilon_pred.flatten(1)) / self.sqrt_αbar[t]
            epsilon_pred = epsilon_pred.flatten(1)
        elif x0_pred is not None:
            x0_pred = x0_pred.flatten(1)
            epsilon_pred = torch.zeros_like(x0_pred)

        # if t <= self.skip_step, then mask is 1, which means return pred_x0
        # otherwise, mask is 0, which means return diffuse
        mask = (t == 0).to(x0_pred.dtype).view(-1, 1)
        output = (x0_pred * mask + self.diffuse(x0_pred, next_t, epsilon_pred) * (1 - mask)).view(*original_shape)

        if need_x0:
            return output, x0_pred.view(*original_shape)
        return output


    @torch.no_grad()
    def denoise(self,
                x_T: Tensor,
                pred_func: Callable[[Tensor, Tensor, Any], Tuple[Tensor, Tensor, Tensor]],
                verbose: bool = False,
                **pred_func_args) -> Tensor:
        """
        Denoises a sample from the final time step T to the initial time step 0.

        :param x_T: Sample at the final time step T.
        :param pred_func: Function to predict x_0 and noise.
        :param verbose: Whether to display a progress bar.
        :param pred_func_args: Additional arguments for the prediction function.
        :return: The denoised sample at time step 0.
        """
        x_t = x_T.clone()
        all_t = torch.arange(self.T, dtype=torch.long, device=self.device).repeat(x_T.shape[0], 1)  # (B, T)

        # [T, T-s, T-2s, ..., k], k >= 0
        t_schedule = list(range(self.T - 1, -1, -self.skip_step))
        if t_schedule[-1] != 0:
            t_schedule.append(0)
        # [T, T-s, T-2s, ..., 0]

        pbar = tqdm(t_schedule) if verbose else t_schedule
        for ti, t in enumerate(pbar):
            x0_pred, epsilon_pred, v_pred = pred_func(x_t, all_t[:, t], **pred_func_args)
            t_next = 0 if ti + 1 == len(t_schedule) else t_schedule[ti + 1]
            x_t = self.denoiseStep(x0_pred, epsilon_pred, v_pred, x_t, all_t[:, t], all_t[:, t_next])

        return x_t


    def combineNoise(self, eps_0_to_t, eps_t_to_tp1, t):
        """
        traj[1] = sqrt(alpha[0]) * traj[0] + sqrt(1 - alpha[0]) * eps_0:1

        traj[t] = sqrt(alpha[t-1]) * traj[t-1] + sqrt(beta[t-1]) * eps_t-1:t
        traj[t] = sqrt(alpha_bar[t-1]) * traj[0] + sqrt(1 - alpha_bar[t-1]) * eps_0:t

        traj[t-1] = sqrt(alpha_bar[t-2]) * traj[0] + sqrt(1 - alpha_bar[t-2]) * eps_0:t-1

        sqrt(alpha[t-1]) * traj[t-1] + sqrt(beta[t-1]) * eps_t-1:t = sqrt(alpha_bar[t-1]) * traj[0] + sqrt(1 - alpha_bar[t-1]) * eps_0:t

        sqrt(alpha[t-1]) * (sqrt(alpha_bar[t-2]) * traj[0] + sqrt(1 - alpha_bar[t-2]) * eps_0:t-1) + sqrt(beta[t-1]) * eps_t-1:t = sqrt(alpha_bar[t-1]) * traj[0] + sqrt(1 - alpha_bar[t-1]) * eps_0:t

        eps_0:t =
        (sqrt(alpha[t-1]) * (sqrt(alpha_bar[t-2]) * traj[0] + sqrt(1 - alpha_bar[t-2]) * eps_0:t-1) + sqrt(beta[t-1]) * eps_t-1:t - sqrt(alpha_bar[t-1]) * traj[0]) / sqrt(1 - alpha_bar[t-1])

        term_1 = sqrt(alpha[t-1]) * sqrt(alpha_bar[t-2]) * traj[0]
        term_2 = sqrt(alpha[t-1]) * sqrt(1 - alpha_bar[t-2]) * eps_0:t-1
        term_3 = sqrt(1 - alpha[t-1]) * eps_t-1:t
        term_4 = sqrt(alpha_bar[t-1]) * traj[0]
        term_5 = sqrt(1 - alpha_bar[t-1])

        eps_0:t = (term_1 + term_2 + term_3 - term_4) / term_5

        term 1 and 4 cancel out, so we can simplify it to:
        eps_0:t = (term_2 + term_3) / term_5 =
        (sqrt(alpha[t-1]) * sqrt(1 - alpha_bar[t-2]) * eps_0:t-1 + sqrt(1 - alpha[t-1]) * eps_t-1:t) / sqrt(1 - alpha_bar[t-1])

        :param eps_0_to_t: Combined noise,  (B, 2, L)
        :param eps_t_to_tp1: Noise for step, (B, 2, L)
        :param t: t int {0, 1, 2, ... T-1}
        :return: eps_0_to_tp1
        """
        if t == 0:
            return eps_t_to_tp1

        original_shape = eps_0_to_t.shape

        term_2 = torch.sqrt(self.alpha[t]) * self.sqrt_1_m_αbar[t - 1] * eps_0_to_t.flatten(1)

        term_3 = torch.sqrt(1 - self.alpha[t]) * eps_t_to_tp1.flatten(1)

        return ((term_2 + term_3) / self.sqrt_1_m_αbar[t]).view(original_shape)


    def extractNoise(self, x_0, x_t, t):
        """
        According to diffuse:
        x_t = sqrt(alpha_bar[t]) * x_0 + sqrt(1 - alpha_bar[t]) * eps_0:t

        eps_0:t = (x_tp1 - sqrt(alpha_bar[t]) * x_0) / sqrt(1 - alpha_bar[t])
        """

        original_shape = x_0.shape

        eps_0_to_t = (x_t.flatten(1) - self.sqrt_αbar[t] * x_0.flatten(1)) / self.sqrt_1_m_αbar[t]

        return eps_0_to_t.view(original_shape)


    def computeVelocity(self, x_0: Tensor, eps_0_to_t: Tensor, t: int) -> Tensor:
        """
        Computes the variance term V for the diffusion process.

        :param x_0: Original sample.
        :param eps_0_to_t: Combined noise.
        :param t: Current time step.
        :return: The variance term V at time step t.
        """
        original_shape = x_0.shape
        return (self.sqrt_αbar[t] * eps_0_to_t.flatten(1) - self.sqrt_1_m_αbar[t] * x_0.flatten(1)).view(original_shape)


