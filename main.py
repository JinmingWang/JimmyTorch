from JimmyExperiment import JimmyExperiment
from Models import SampleCNN
from Datasets import DEVICE
import pandas as pd
import os


def train():
    """
    Train the SampleCNN model on MNIST dataset.
    Returns the trainer object.
    """
    # Create experiment with custom directory name
    experiment = JimmyExperiment(
        comments="Training SampleCNN on MNIST",
        dir_name="sample_cnn_baseline"
    )
    
    # Configure experiment
    experiment.model_cfg.cls = SampleCNN
    experiment.dataset_cfg.set_name = "train"
    experiment.constants["n_epochs"] = 10
    experiment.constants["eval_interval"] = 1
    experiment.constants["early_stop_lr"] = 1e-6
    
    # Start training
    trainer = experiment.start()
    
    return trainer


def test(run_folder: str, set_name: str = "test") -> None:
    """
    Test the trained model on a dataset.
    
    :param run_folder: Path to the run folder containing the checkpoint
    :param set_name: Dataset split to test on (default: "test")
    """
    # Load model
    model = SampleCNN()
    model.loadFrom(os.path.join(run_folder, "best.pth"))
    model.to(DEVICE)
    model.eval()
    
    # Create experiment for testing
    experiment = JimmyExperiment(
        comments=f"Testing on {set_name} set"
    )
    experiment.dataset_cfg.set_name = set_name
    
    # Build test dataset
    test_set = experiment.dataset_cfg.build()
    
    # Run test
    test_report = experiment.test(model, test_set)
    
    # Save results
    output_path = os.path.join(run_folder, f"test_report_{set_name}.csv")
    test_report.to_csv(output_path, index=False)
    print(f"Test results saved to: {output_path}")
    
    # Print summary
    print(f"\nTest Results Summary:")
    print(f"  Mean Loss: {test_report['Eval/Main'].mean():.4f}")
    print(f"  Std Loss:  {test_report['Eval/Main'].std():.4f}")


if __name__ == '__main__':
    # Training
    print("="*50)
    print("Starting Training")
    print("="*50)
    trainer = train()
    
    # Extract run folder path
    run_folder = trainer.save_dir
    
    # Testing
    print("\n" + "="*50)
    print("Starting Testing")
    print("="*50)
    test(run_folder, set_name="test")
    
    # Optional: test on other splits
    # test(run_folder, set_name="eval")