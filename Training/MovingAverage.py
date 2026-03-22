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

    def toList(self) -> list[float]:
        if self.count == self.window_size:
            return (torch.cat((self.values[self.idx:], self.values[:self.idx]))).tolist()
        else:
            return (self.values[:self.count]).tolist()


class MovingAvgGroup():
    def __init__(self, item_names: list[str], window_sizes: int | list[int]):
        if isinstance(window_sizes, int):
            window_sizes = [window_sizes] * len(item_names)
        assert len(item_names) == len(window_sizes)
        self.item_names = item_names
        self.window_sizes = window_sizes
        self.avgs = {name: MovingAvg(size) for name, size in zip(item_names, window_sizes)}

    def update(self, values: list[float] | dict[str, float]):
        if isinstance(values, dict):
            for name in values:
                assert name in self.item_names, f"Invalid item name: {name}"
                self.avgs[name].update(values[name])
        else:
            for name, value in zip(self.item_names, values):
                self.avgs[name].update(value)


    def get(self) -> dict[str, float]:
        return {name: self.avgs[name].get() for name in self.item_names}
