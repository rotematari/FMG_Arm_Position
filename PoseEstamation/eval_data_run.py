from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
from data_processes.data_loader import FMGPoseDataset, FMGPoseDataset_data_importance
from utils.utils import load_yaml_config ,set_device, set_seed
from train.train import train_for_data_test
from evaluation.evaluate import test_model
from models.get_model import get_model
import os 


def train_for_one_dataset(config,trainDataset,testDataset,device,number_of_data_samples,experiment):

    # Create the dataloader
    trainDataloader = DataLoader(trainDataset, batch_size=config["batch_size"], shuffle=True,drop_last=True)
    testDataloader = DataLoader(testDataset, batch_size=config["batch_size"], shuffle=False,drop_last=True)
    # Create the model

    model = get_model(config=config)

    model = model.to(device=device) 
    # Create the optimizer
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    # Create the loss function
    # criterion = torch.nn.MSELoss()
    # Train the model

    train_for_data_test(config,model,optimizer, trainDataloader,testDataloader,trainDataset.label_scalar, 
                        device=device,wandb_run=None,number_of_data_samples=number_of_data_samples,experiment=experiment)

if __name__ == '__main__':
    # Load the config file
    config = load_yaml_config('config.yaml')
    trainDataset_for_scaler = FMGPoseDataset(config,mode='train')
    testDataset = FMGPoseDataset(config,mode='test',feature_scalar=trainDataset_for_scaler.feature_scalar,label_scalar= trainDataset_for_scaler.label_scalar)
    # Set the seed
    set_seed(config["seed"])
    device = set_device()
    # Example list of datasets
    train_datasets = []
    # Get all files in the directory
    all_files = [file for file in os.listdir(config["train_data_path"]) if file.endswith('.csv')]
    for number_of_data_samples in range(len(all_files)):
        for experiment in range(config["num_of_data_sets_experiments"]):
            print(f"{number_of_data_samples} data samples experiment :{experiment}\n")
            trainDataset = FMGPoseDataset_data_importance(config,mode='train',number_of_data_samples=number_of_data_samples + 1)
            # train_datasets.append(trainDataset)
            train_for_one_dataset(config,trainDataset,testDataset,device,number_of_data_samples+1,experiment)

