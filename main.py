from JimmyExperiment import JimmyExperiment

if __name__ == '__main__':
    for lr in [1e-4, 2e-4, 5e-4, 1e-3]:
        # Create a new experiment with custom directory name
        experiment = JimmyExperiment(
            comments=f"Trying with peak_lr={lr}.",
            dir_name=f"peak_lr_{lr}"
        )
        # Specify the key values for this experiment
        experiment.lr_scheduler_cfg.peak_lr = lr
        experiment.dataset_cfg.set_name = "debug"
        experiment.constants["n_epochs"] = 10
        experiment.constants["eval_interval"] = 1
        
        # Start experiment with reasonable comments
        experiment.start()