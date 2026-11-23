"""Launch with torchrun; use main.py for the unchanged legacy example."""
import os
from pathlib import Path
from torch.distributed.elastic.multiprocessing.errors import record
from Distributed import DistributedJimmyExperiment
from Distributed.Sample import DistributedMNISTDataset


@record
def main():
    experiment = DistributedJimmyExperiment("Distributed SampleCNN on MNIST")
    experiment.constants.update(n_epochs=10, early_stop_lr=1e-6,
                                progress_host=os.environ.get("JIMMY_DASHBOARD_HOST", "127.0.0.1"))
    experiment.distributed_cfg["data_prepare_scope"] = "per_node"
    trainer = experiment.start()
    if trainer.distributed_context.is_primary:
        trainer.model.loadFrom(str(Path(trainer.save_dir) / "best.pth"))
        test_set = DistributedMNISTDataset("test", batch_size=64)
        experiment.test(trainer.model, test_set).to_csv(Path(trainer.save_dir) / "test_report_test.csv", index=False)


if __name__ == "__main__":
    main()
