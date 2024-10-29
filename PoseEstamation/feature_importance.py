from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
from data_processes.data_loader import FMGPoseDatasetFeatureImportance
from utils.utils import load_yaml_config ,set_device, set_seed
from train.train import train
from evaluation.evaluate import test_model
from models.get_model import get_model
from os.path import join
import os 
import csv


def save_experiment_results(file_path, experiment_data):
    """
    Save the results of the experiment to a CSV file.
    
    Parameters:
    - file_path: The path to the CSV file.
    - experiment_data: A dictionary containing the experiment data.
    """
    # Define CSV headers (make sure the order of fields is the same as the keys in experiment_data)
    headers = [
        'sensor_to_shufle', 'rmse_loss', 
        'wrist_error','wrist_std', 
        'avg_elbow_error','elbow_std'
    ]

    # Check if the file exists
    file_exists = os.path.isfile(file_path)

    # Open the file in append mode
    with open(file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)

        # If the file does not exist, write the header
        if not file_exists:
            writer.writeheader()

        # Write the experiment data to the CSV
        writer.writerow(experiment_data)

if __name__ == '__main__':
    # Load the config file
    config = load_yaml_config('config.yaml')
    # Set the seed
    set_seed(config["seed"])
    device = set_device()

    # Load the dataset
    trainDataset = FMGPoseDatasetFeatureImportance(config,mode='train')
    model = get_model(config=config)

    model = model.to(device=device) 
    # Create the optimizer
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    for sensor_to_shufle in config["feature_index"]:
        print(f"sensor_to_shufle -> {sensor_to_shufle}")
        testDataset = FMGPoseDatasetFeatureImportance(config,mode='test',
                                                      feature_scalar=trainDataset.feature_scalar,
                                                      label_scalar= trainDataset.label_scalar,
                                                      sensor_to_shufle=sensor_to_shufle)

        testDataloader = DataLoader(testDataset, batch_size=config["batch_size"], shuffle=False,drop_last=True)
        # Create the model
        avg_loss,avg_critic_loss,avg_iter_time, avg_location_error,avg_euc_end_effector_error,max_euc_end_effector_error,R2_score,avg_elbow_error,wrist_std,elbow_std = test_model(
                model=model,
                config=config,
                data_loader=testDataloader,
                label_scaler=trainDataset.label_scalar,
                device=device,
                task='test',
                make_pdf= True,
            )
        # Prepare experiment data to save to CSV
        experiment_data = {
            'sensor_to_shufle': sensor_to_shufle,
            'rmse_loss': round(avg_loss, 5),
            'wrist_error': round(avg_euc_end_effector_error, 5),
            'wrist_std': round(wrist_std, 5),
            'avg_elbow_error': round(avg_elbow_error, 5),
            'elbow_std': round(elbow_std, 5),
        }
        # Save results to CSV
        save_experiment_results(join(config["data_test_path"],"csv_feature_importance_zero_results.csv"), experiment_data)


