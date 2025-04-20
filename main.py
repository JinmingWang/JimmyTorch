from JimmyExperiment import JimmyExperiment

if __name__ == '__main__':
    for lr in [1e-4, 2e-4, 5e-4, 1e-3]:
        # Create a new experiment
        experiment = JimmyExperiment()
        # Specify the key values for this experiment
        experiment.hyper_params["lr_scheduler_args"]["peak_lr"] = lr
        # Start experiment with reasonable comments
        experiment.start(comments=f"Trying with peak_lr={lr}.")