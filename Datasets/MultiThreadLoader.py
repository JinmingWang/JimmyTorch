# Implementation of a multi-threaded data loader for JimmyDataset
# It simply uses the __getitem__ function of JimmyDataset to get the data
# which will get a dictionary as output
# It uses a fixed size queue to store the data
# While the model is training, it will continuously get data from the queue
# So the training pipeline will not be blocked by the data loading


import threading
import queue
from typing import Iterable
from .JimmyDataset import JimmyDataset

class MultiThreadLoader:

    """
    Multithreaded data loader for JimmyDataset.

    Usage:

    loader = MultiThreadLoader(dataset, queue_size=3)
    for data_dict in loader:
        # Process data_dict
        pass

    """
    def __init__(self, dataset: JimmyDataset, queue_size: int = 3):
        """
        Multi-threaded data loader for JimmyDataset
        :param queue_size: size of the queue
        :param dataset: dataset to load data from
        """
        self.queue_size = queue_size
        self.dataset = dataset
        self.queue = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._load_data, daemon=True)
        self.thread.start()


    def __len__(self):
        return self.dataset.n_batches


    def __iter__(self) -> Iterable:
        for _ in range(len(self)):
            if self.stop_event.is_set():
                break
            data_dict = self.queue.get()
            yield data_dict
            self.queue.task_done()


    def _load_data(self):
        """
        Load data from the dataset and put it in the queue
        :return:
        """
        for data_dict in self.dataset:
            if self.stop_event.is_set():
                break
            self.queue.put(data_dict)


    def queueSize(self) -> int:
        """
        Get the size of the queue
        :return: size of the queue
        """
        return self.queue.qsize()