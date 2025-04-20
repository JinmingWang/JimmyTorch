import torch
import math

# Enable interactive mode for matplotlib
# import matplotlib.pyplot as plt
# plt.ion()

class JimmyLRScheduler:
    """
    Jimmy Learning Rate Scheduler is a combination of multiple learning rate scheduler designs:
    1. ReduceLROnPlateau: Reduce learning rate when a metric has stopped improving.
    2. CosineAnnealingLR: Set the learning rate to follow a cosine curve.
    3. OneCycleLR: Set the learning rate to follow a one-cycle policy.
    4. ExponentialLR: Set the learning rate to follow an exponential decay.

    More precisely, Jimmy LR Scheduler does this:
    1. A warming up phase, where the LR increases from 0 to the peak learning rate.
    2. A cosine annealing phase, where the LR follows a cosine curve.
    3. A decay phase, when the metric has stopped improving, the LR slowly decays until the metric improves again.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 init_lr: float,
                 peak_lr: float,
                 min_lr: float,
                 warmup_count: int,
                 window_size: int = 10,
                 patience: int = 10,
                 decay_rate: float = 0.9,
                ) -> None:

        self.optimizer = optimizer
        self.init_lr = init_lr
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.warmup_count = warmup_count
        self.window_size = window_size
        self.patience = patience
        self.decay_rate = decay_rate

        self.n_iter = 0
        self.lr_without_cos = init_lr

        self.patience_count = 0
        self.metric_list = []
        # self.figure, self.ax = plt.subplots()  # Create a figure and axis for dynamic plotting


    def phaseWarmUP(self) -> float:
        """
        Phase A: Warming up phase, where the LR increases from 0 to the peak learning rate.
        :return:
        """
        self.lr_without_cos = self.peak_lr * math.sin(math.pi / (2 * self.warmup_count) * self.n_iter) ** 2
        return self.lr_without_cos


    def phaseCosine(self) -> float:
        """
        Phase B: Cosine annealing phase, where the LR follows a cosine curve.
        :return:
        """
        return self.lr_without_cos * (0.05 * math.cos(10 * (self.n_iter / self.warmup_count - 1)) ** 2 + 0.95)



    def phaseDecay(self) -> float:
        """
        Phase C: Decay phase, when the metric has stopped improving, the LR slowly decays until the metric improves again.
        :return:
        """
        self.lr_without_cos = max(self.min_lr, self.lr_without_cos * self.decay_rate)
        return self.phaseCosine()


    def update(self, metric: float) -> None:
        """
        Update the learning rate based on the current iteration.
        :param metric: The current metric to check for improvement.
        :return: The updated learning rate.
        """
        self.metric_list.append(metric)
        self.n_iter += 1
        if self.n_iter < self.warmup_count:
            lr = self.phaseWarmUP()
        elif self.isImproving():
            lr = self.phaseCosine()
        else:
            lr = self.phaseDecay()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


    def isImproving(self) -> bool:
        """
        Check if the metric is improving. Say, the metric is the loss.

        We store a set of losses, denoted L = {L1, L2, ..., Ln}.
        We can fit a line to the metrics, which is the first derivative of the losses, denoted as dL.
        dL is loss change speed, negative dL means decreasing loss.
        We can also compute the second derivative of the losses, denoted as d2L.
        d2L is the loss change acceleration.
        - dL and - d2L: decreasing and concave down loss. ↴
        - dL and + d2L: decreasing and concave up loss. ↳
        + dL and - d2L: increasing and concave down loss. ↷
        + dL and + d2L: increasing and concave up loss. ↗

        We define dL > -eps_1 as non-improving. In this case, loss is increasing or no change.
        We define d2L > -eps_2 as non-waitable. Because we see no sign of improvement.

        :param loss: The current loss.
        :return: True if the loss is improving, False otherwise.
        """
        if len(self.metric_list) <= self.window_size:
            return True
        else:
            self.metric_list.pop(0)

        # Center the metrics
        metrics = torch.tensor(self.metric_list)
        metric_min = metrics.min()
        metric_max = metrics.max()
        metrics = (metrics - metric_min) / (metric_max - metric_min)
        # metrics is now in the range [0, 1]

        # Compute linear regression
        x = torch.linspace(0, 1, self.window_size)
        A = torch.vstack([x, torch.ones(self.window_size)]).T.view(1, -1, 2)
        c1, c0 = torch.linalg.lstsq(A, metrics.view(1, -1, 1)).solution[0]
        dL = c1[0].item()

        # self.ax.clear()  # Clear the previous plot
        # self.ax.plot(x.numpy(), metrics.numpy(), label='Metrics')
        # self.ax.plot(x.numpy(), c1[0].item() * x.numpy() + c0[0].item(), label='Linear Fit')

        # Compute second derivative using quadratic regression
        A = torch.vstack([x ** 2, x, torch.ones(len(x))]).T.view(1, -1, 3)
        c2, c1, c0 = torch.linalg.lstsq(A, metrics.view(1, -1, 1)).solution[0]
        d2L = c2[0].item()

        # Update the plot dynamically
        # self.ax.plot(x.numpy(), c2[0].item() * x.numpy() ** 2 + c1[0].item() * x.numpy() + c0[0].item(), label='Quadratic Fit')
        # self.ax.set_title(f"Learning Rate Scheduler {self.patience_count}")
        # self.ax.set_xlabel("Epoch")
        # self.ax.set_ylabel("Metrics")
        # self.ax.legend()
        # self.figure.canvas.draw()  # Redraw the figure
        # self.figure.canvas.flush_events()  # Flush the GUI events

        if dL > -0.001 and d2L > -0.001:
            self.patience_count += 1

        if self.patience_count > self.patience:
            self.patience_count = 0
            return False

        return True
