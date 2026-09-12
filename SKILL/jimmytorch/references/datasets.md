# Dataset contracts

Read this reference when creating or modifying a dataset, its batch dictionary, or queued loading.

## Required subclass behavior

`JimmyDataset(set_name, batch_size, drop_last=False, shuffle=False)` is an iterable over batches. A subclass must:

1. Call `super().__init__(set_name, batch_size, drop_last, shuffle)`.
2. Load/select the requested split.
3. Set `self.n_samples` to the selected sample count.
4. Implement `__getitem__(batch_idx)` for a zero-based **batch** index.
5. Return a dictionary whose stable, descriptive keys match model expectations.

The usual indexing pattern is:

```python
start = batch_idx * self.batch_size
end = min(start + self.batch_size, self.n_samples)
indices = self._indices[start:end]
return {"features": self.features[indices], "labels": self.labels[indices]}
```

`_indices` is created by `__iter__`; direct calls to a custom `__getitem__` may therefore require `iter(dataset)` first. `len(dataset)` returns samples. `dataset.n_batches` returns batches and accounts for `drop_last`.

## Split and device decisions

The recognized convention is `train`, `eval`, `test`, `debug`, and `all`, but the base class only stores the supplied name. Keep all split logic deterministic and shared across training and testing.

`Datasets.DEVICE` chooses CUDA when available and CPU otherwise. Built-in examples preload onto it, but preloading is not enforced by the base class. For large data, choose a bounded strategy and ensure returned tensors match the model device.

## MultiThreadLoader

`MultiThreadLoader(dataset, queue_size=3)` creates one daemon producer thread and a bounded queue. It does not accept `num_workers`. Use it only when overlapping `__getitem__` work is useful; keep that work thread-safe and make sure producer exceptions cannot leave consumers waiting indefinitely.

## Validation

- Confirm split sizes and boundary batches for both `drop_last` values.
- Confirm shuffling changes order without changing membership.
- Validate every dictionary key, dtype, shape, and device used by the model.
- Exercise `n_samples < batch_size`, an exact multiple, and a remainder batch.
- Avoid full dataset downloads in routine tests; use synthetic or tiny fixture tensors.
