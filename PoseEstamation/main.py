from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
from data_processes.data_loader import FMGPoseDataset
from utils.utils import load_yaml_config ,set_device, set_seed
from train.train import train
from evaluation.evaluate import test_model
from models.get_model import get_model
if __name__ == '__main__':
    # Load the config file
    config = load_yaml_config('config.yaml')
    # Set the seed
    set_seed(config["seed"])
    device = set_device()

    # Load the dataset
    trainDataset = FMGPoseDataset(config,mode='train')
    testDataset = FMGPoseDataset(config,mode='test',feature_scalar=trainDataset.feature_scalar,label_scalar= trainDataset.label_scalar)
    print(f"Train Data set length {len(trainDataset)}")
    print(f"Test Data set length {len(testDataset)}")

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
    if not config["pre_trained"]:
        train(config,model,optimizer, trainDataloader,testDataloader,trainDataset.label_scalar,trainDataset.feature_scalar, device=device,wandb_run=None)
    else:
        avg_loss,avg_critic_loss,avg_iter_time, avg_location_error,avg_euc_end_effector_error,max_euc_end_effector_error,R2_score,avg_elbow_error,wrist_std,elbow_std = test_model(
                model=model,
                config=config,
                data_loader=testDataloader,
                label_scaler=trainDataset.label_scalar,
                device=device,
                task='test',
                make_pdf= True,
            )