import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torch

from utils.utils import load_yaml_config ,set_device, set_seed
from tune_model.ray_train_fe import train_model
import json
from datetime import datetime
import os


def main(score=1):
    # Load the config file
    my_config = load_yaml_config('config.yaml')
    # Set the seed
    set_seed(my_config["seed"])
    device = set_device()



    search_space = {
        "name": "TransformerModel",
        "dropout": tune.quniform(0.0900, 0.2000, 0.001),
        "learning_rate": tune.qloguniform(1e-4, 1e-2, 1e-4),
        "weight_decay": tune.qloguniform(1e-4, 1e-2, 1e-4),
        "warmup_length": tune.quniform(0.3000, 0.4000, 0.001),
        "batch_size": tune.choice([32]),
        "sequence_length": tune.choice([128]),
        "epochs": 15,
    }


    # Extract the maximum number of epochs from your search space
    max_epochs = 15
    # Use an ASHA scheduler for efficient hyperparameter optimization
    scheduler = ASHAScheduler(
        metric="wrist_error",
        mode="min",
        max_t=max_epochs,
        grace_period=9,
        reduction_factor=2
    )

    # Define a progress reporter
    reporter = CLIReporter(
        metric_columns=["loss","wrist_error","wrist_std_eror","elbow_error","elbow_std_eror","best_epoch" ,"training_iteration"],
        sort_by_metric=True
    )

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Run Ray Tune
    result = tune.run(
        tune.with_parameters(train_model, my_config=my_config,device=device,score=score),
        resources_per_trial={"cpu": 20, "gpu": int(torch.cuda.is_available())},
        config=search_space,
        num_samples=100,  # Number of hyperparameter configurations to try
        scheduler=scheduler,
        progress_reporter=reporter,
        raise_on_failed_trial=False  # Add this parameter
    )

    # Get the best trial
    best_trial = result.get_best_trial("wrist_error", "min", "all")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final wrist error: {best_trial.last_result['wrist_error']}")
    print(f"Best trial best at epoch: {best_trial.last_result['best_epoch']}")
    # Access the best trial's configuration
    best_config = best_trial.config

    # Access the last reported loss and wrist error
    best_loss = float(best_trial.last_result["loss"])
    best_wrist_error = float(best_trial.last_result["wrist_error"])
    best_epoch = int(best_trial.last_result["best_epoch"])
    # Save the best trial's configuration and metrics
    best_trial_info = {
        "config": best_config,
        "loss": best_loss,
        "wrist_error": best_wrist_error,
        "wrist_std_error": float(best_trial.last_result["wrist_std_eror"]),
        "elbow_error": float(best_trial.last_result["elbow_eror"]),
        "elbow_std_error": float(best_trial.last_result["elbow_std_eror"]),
        "best_epoch": best_epoch
    }
    # Get the current date and time
    now = datetime.now()

    # Format the date as month_day
    month_day_str = now.strftime("%m_%d")

    # Format the date and time as month_day_hour_min
    date_time_str = now.strftime("%m_%d_%H_%M")
    # Create the directory path
    directory = os.path.join("ray_tune_results_fe", str(score),month_day_str)
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)
    # Create the filename with the date and time appended
    filename = f"best_trial_info_{date_time_str}.json"
    # Combine directory and filename to get the full file path
    filepath = os.path.join(directory, filename)
    with open(filepath, "w") as f:
        json.dump(best_trial_info, f, indent=4)

    print("Best trial configuration saved to 'best_trial_info.json'")
    print(f"Best trial config: {best_config}")
    print(f"Best trial final validation loss: {best_loss}")
    print(f"Best trial final wrist error: {best_wrist_error}")
    # Optionally, you can retrieve the best model and use it for inference

if __name__ == '__main__':

    for score in [0.5,1.0,1.5,2.0,2.5]:
        main(score=score)