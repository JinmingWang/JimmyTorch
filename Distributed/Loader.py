"""Bounded batch prefetching with exception propagation and explicit shutdown."""
import queue
import threading
import torch
from .Dataset import DistributedJimmyDataset


class DistributedLoader:
    def __init__(self, dataset: DistributedJimmyDataset, queue_size: int = 3, device=None):
        if queue_size < 1:
            raise ValueError("queue_size must be positive")
        self.dataset = dataset
        self.device = device
        self.queue = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loadData, daemon=True)
        self.thread.start()

    def __len__(self):
        return self.dataset.n_batches

    def __iter__(self):
        try:
            while True:
                kind, value = self.queue.get()
                if kind == "end":
                    return
                if kind == "error":
                    raise value
                yield value
        finally:
            self.close()

    def _put(self, item):
        while not self.stop_event.is_set():
            try:
                self.queue.put(item, timeout=0.1)
                return
            except queue.Full:
                pass

    def _loadData(self):
        try:
            if self.device is not None and self.device.type == "cuda":
                torch.cuda.set_device(self.device)
            for data_dict in self.dataset:
                if self.stop_event.is_set():
                    break
                self._put(("data", data_dict))
        except BaseException as error:
            self._put(("error", error))
        finally:
            self._put(("end", None))

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=2.0)

    def queueSize(self):
        return self.queue.qsize()
