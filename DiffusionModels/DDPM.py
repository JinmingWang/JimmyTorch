import torch
from typing import *
from tqdm import tqdm
from math import log

Tensor = torch.Tensor


class DDPM:
    """
    DDPM (Denoising Diffusion Probabilistic Models) class for diffusion-based generative modeling.

    Attributes:
        min_beta (float): Minimum beta value for the diffusion process.
        max_beta (float): Maximum beta value for the diffusion process.
        max_diffusion_step (int): Total number of diffusion steps.
        device (str): Device to perform computations on ('cuda' or 'cpu').
        scale_mode (Literal["linear", "quadratic", "log"]): Mode for scaling beta values.
    """
    def __init__(self,
                 min_beta: float = 0.0001,
                 max_beta: float = 0.002,
                 max_diffusion_step: int = 100,
                 device: str = 'cuda',
                 scale_mode: Literal["linear", "quadratic", "log"] = "linear"):
        """
        Initializes the DDPM model with the given parameters.

        :param min_beta: Minimum beta value for the diffusion process.
        :param max_beta: Maximum beta value for the diffusion process.
        :param max_diffusion_step: Total number of diffusion steps.
        :param device: Device to perform computations on ('cuda' or 'cpu').
        :param scale_mode: Mode for scaling beta values.
        """
        if scale_mode == "quadratic":
            betas = torch.linspace(min_beta ** 0.5, max_beta ** 0.5, max_diffusion_step).to(device) ** 2
        elif scale_mode == "log":
            betas = torch.exp(torch.linspace(log(min_beta), log(max_beta), max_diffusion_step).to(device))
        else:
            betas = torch.linspace(min_beta, max_beta, max_diffusion_step).to(device)

        self.device = device

        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.T = max_diffusion_step

        self.betas = betas.view(-1, 1)
        self.alphas = alphas.view(-1, 1)
        self.alpha_bars = alpha_bars.view(-1, 1)
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars).view(-1, 1)
        self.sqrt_1m_alpha_bars = torch.sqrt(1 - alpha_bars).view(-1, 1)

        # Backward-compatible aliases used by older call sites.
        self.beta = self.betas
        self.alpha = self.alphas
        self.αbar = self.alpha_bars
        self.sqrt_αbar = self.sqrt_alpha_bars
        self.sqrt_1_m_αbar = self.sqrt_1m_alpha_bars

    def diffuseStep(self, x_t: Tensor, t: Union[int, Tensor], epsilon_t_to_tp1: Tensor) -> Tensor:
        """
        Diffuse one step from x_t to x_{t+1}.

        :param x_t: Sample at timestep t.
        :param t: Timestep, either int or Tensor of shape (B,).
        :param epsilon_t_to_tp1: Noise to add.
        :return: Noisy sample at timestep t+1.
        """
        original_shape = x_t.shape
        x_t_flat = x_t.flatten(1)
        epsilon_flat = epsilon_t_to_tp1.flatten(1)
        alphas = self.alphas.to(x_t.device)
        sqrt_alpha_t = torch.sqrt(alphas[t])
        sqrt_1m_alpha_t = torch.sqrt(1 - alphas[t])
        x_tp1_flat = sqrt_alpha_t * x_t_flat + sqrt_1m_alpha_t * epsilon_flat
        return x_tp1_flat.view(original_shape)

    def diffuse(self, x_0: Tensor, t: Union[int, Tensor], noise: Tensor = None) -> Tensor:
        """
        Forward diffusion from x_0 to x_t.

        :param x_0: Initial sample.
        :param t: Target timestep.
        :param noise: Gaussian noise, sampled if None.
        :return: Diffused sample at timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        original_shape = x_0.shape
        x_0_flat = x_0.flatten(1)
        noise_flat = noise.flatten(1)
        sqrt_alpha_bars = self.sqrt_alpha_bars.to(x_0.device)
        sqrt_1m_alpha_bars = self.sqrt_1m_alpha_bars.to(x_0.device)
        x_t_flat = sqrt_alpha_bars[t] * x_0_flat + sqrt_1m_alpha_bars[t] * noise_flat
        return x_t_flat.view(original_shape)

    def computeVelocity(self, x_0: Tensor, epsilon: Tensor, t: Union[int, Tensor]) -> Tensor:
        """
        Compute velocity target from x_0 and epsilon.
        """
        original_shape = x_0.shape
        x_0_flat = x_0.flatten(1)
        epsilon_flat = epsilon.flatten(1)
        sqrt_alpha_bars = self.sqrt_alpha_bars.to(x_0.device)
        sqrt_1m_alpha_bars = self.sqrt_1m_alpha_bars.to(x_0.device)
        v = sqrt_alpha_bars[t] * epsilon_flat - sqrt_1m_alpha_bars[t] * x_0_flat
        return v.view(original_shape)

    def _parse_prediction(self, x_t: Tensor, t: Union[int, Tensor], x0_pred=None, epsilon_pred=None, v_pred=None):
        sqrt_alpha_bars = self.sqrt_alpha_bars.to(x_t.device)
        sqrt_1m_alpha_bars = self.sqrt_1m_alpha_bars.to(x_t.device)
        betas = self.betas.to(x_t.device)
        alphas = self.alphas.to(x_t.device)
        alpha_bars = self.alpha_bars.to(x_t.device)

        if v_pred is not None:
            x0_pred_flat = sqrt_alpha_bars[t] * x_t.flatten(1) - sqrt_1m_alpha_bars[t] * v_pred.flatten(1)
            epsilon_pred_flat = sqrt_1m_alpha_bars[t] * x_t.flatten(1) + sqrt_alpha_bars[t] * v_pred.flatten(1)
        elif epsilon_pred is not None:
            epsilon_pred_flat = epsilon_pred.flatten(1)
            x0_pred_flat = (x_t.flatten(1) - sqrt_1m_alpha_bars[t] * epsilon_pred_flat) / sqrt_alpha_bars[t]
        elif x0_pred is not None:
            x0_pred_flat = x0_pred.flatten(1)
            epsilon_pred_flat = (x_t.flatten(1) - sqrt_alpha_bars[t] * x0_pred_flat) / sqrt_1m_alpha_bars[t]
        else:
            raise ValueError('Must provide x0_pred, epsilon_pred, or v_pred')
        return x0_pred_flat, epsilon_pred_flat

    def denoiseStep(self,
                    x_t: Tensor,
                    t: Union[int, Tensor],
                    t_prev: Union[int, Tensor],
                    x0_pred: Tensor = None,
                    epsilon_pred: Tensor = None,
                    v_pred: Tensor = None,
                    need_x0: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Single reverse step from x_t to x_tprev.
        """
        original_shape = x_t.shape
        x0_pred_flat, epsilon_pred_flat = self._parse_prediction(x_t, t, x0_pred, epsilon_pred, v_pred)

        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bars[t]
        alpha_bar_prev = self.alpha_bars[t_prev]

        mu = (x_t.flatten(1) - beta_t / self.sqrt_1m_alpha_bars[t] * epsilon_pred_flat) / torch.sqrt(alpha_t)

        if isinstance(t_prev, int):
            is_final = (t_prev == 0)
        else:
            is_final = (t_prev == 0)
            if is_final.ndim > 0:
                is_final = is_final.view(-1, 1)

        if isinstance(is_final, bool):
            if is_final:
                x_prev_flat = x0_pred_flat
            else:
                sigma = torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * beta_t)
                x_prev_flat = mu + sigma * torch.randn_like(mu)
        else:
            sigma = torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * beta_t)
            x_prev_noised = mu + sigma * torch.randn_like(mu)
            x_prev_flat = is_final * x0_pred_flat + (~is_final) * x_prev_noised

        x_prev = x_prev_flat.view(original_shape)
        if need_x0:
            return x_prev, x0_pred_flat.view(original_shape)
        return x_prev

    @torch.no_grad()
    def denoise(self,
                x_T: Tensor,
                pred_func: Callable[[Tensor, Tensor, Any], Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]],
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
        batch_size = x_T.shape[0]
        device = x_T.device
        timesteps = list(range(self.T - 1, -1, -1))
        iterator = tqdm(timesteps, desc='DDPM sampling') if verbose else timesteps
        for i, t in enumerate(iterator):
            t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=device)
            x0_pred, epsilon_pred, v_pred = pred_func(x_t, t_tensor, **pred_func_args)
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            t_prev_tensor = torch.full((batch_size,), t_prev, dtype=torch.long, device=device)
            x_t = self.denoiseStep(x_t, t_tensor, t_prev_tensor, x0_pred, epsilon_pred, v_pred)

        return x_t


    def combineNoise(self, eps_0_to_t, eps_t_to_tp1, t):
        """

        :param eps_0_to_t: Combined noise,  (B, 2, L)
        :param eps_t_to_tp1: Noise for step, (B, 2, L)
        :param t: t int {0, 1, 2, ... T-1}
        :return: eps_0_to_tp1
        """
        if t == 0:
            return eps_t_to_tp1

        original_shape = eps_0_to_t.shape

        term_1 = torch.sqrt(self.alpha[t]) * self.sqrt_1_m_αbar[t - 1] * eps_0_to_t.flatten(1)

        term_2 = torch.sqrt(1 - self.alpha[t]) * eps_t_to_tp1.flatten(1)

        return ((term_1 + term_2) / self.sqrt_1_m_αbar[t]).view(original_shape)
