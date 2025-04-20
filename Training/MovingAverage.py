import torch


class MovingAvg():
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.values = torch.zeros(window_size, dtype=torch.float32, device='cuda')
        self.idx = 0
        self.count = 0

    def update(self, value: float):
        self.values[self.idx] = value
        self.idx = (self.idx + 1) % self.window_size
        self.count = min(self.count + 1, self.window_size)

    def get(self) -> float:
        return self.values[:self.count].mean().item()

    def __len__(self):
        return self.count
