import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import AdamW
from models.get_model import get_model
import ray
from ray import tune
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from utils.metrics import RMSELoss, TSS, RSS, R2
from evaluation.evaluate import test_model
import time
from torch.nn import MSELoss
from data_processes.data_loader import FMGPoseDataset
from ray.train import report
import numpy as np 
def update_config(config, my_config):
    """
    Updates my_config with values from config for keys that are present in config.

    Args:
        config (dict): Dictionary containing parameters to be updated (e.g., hyperparameters from Ray Tune).
        my_config (dict): Original configuration dictionary to be updated.

    Returns:
        dict: Updated my_config dictionary.
    """
    for key in config:
        my_config[key] = config[key]
    
    print(my_config)
    return my_config
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, yhat, y):
        # Ensure inputs are tensors, not tuples
        if isinstance(yhat, tuple):
            yhat = yhat[0]  # Take first element if it's a tuple
        if isinstance(y, tuple):
            y = y[0]  # Take first element if it's a tuple
            
        return torch.sqrt(self.mse(yhat, y))
    
def train(config, model, optimizer, train_loader, val_loader, scalar, device='cpu', wandb_run=None):
    num_epochs = config["epochs"]
    warmup_length = config["warmup_length"]
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = int(0.2 * num_training_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps,
        num_cycles=0.5
    )

    criterion = RMSELoss()
    train_losses = []
    val_losses = []
    avg_euc_end_effector_errors = []
    TSS_losses = []
    RSS_losses = []
    best_val_loss = 10

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_iterator = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")
        
        for i, (inputs, targets) in train_iterator:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            # Forward pass and get predictions
            elbow_pred, wrist_pred, fmg = model(inputs)
            
            # Split targets into elbow and wrist components
            elbow_target = targets[:, :3]
            wrist_target = targets[:, 3:6]

            # Calculate losses separately
            elbow_loss = criterion(elbow_pred, elbow_target)
            wrist_loss = criterion(wrist_pred, wrist_target)
            
            # Combine losses
            loss = elbow_loss + wrist_loss

            loss.backward()
            train_loss += loss.item()

            # Calculate metrics
            combined_pred = torch.cat([elbow_pred, wrist_pred], dim=1)
            TSS_losses.append(TSS(targets.cpu().detach().numpy()))
            RSS_losses.append(RSS(targets.cpu().detach().numpy(), combined_pred.cpu().detach().numpy()))

            optimizer.step()
            if scheduler:
                scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']
            train_iterator.set_description(
                f"Epoch [{epoch}/{num_epochs}] Train Loss: {(train_loss/(i+1)):.4f} "
                f"R^2:{R2(TSS_losses,RSS_losses):.3f} LR: {current_lr:.6f}"
            )

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        val_loss,avg_critic_loss,avg_iter_time, avg_location_error,avg_euc_end_effector_error,max_euc_end_effector_error,R2_score,avg_elbow_error,wrist_std,elbow_std = test_model(
            model=model,
            config=config,
            data_loader=val_loader,
            label_scaler=scalar,
            device=device,
            epoch=epoch,
            task='validate',
            make_pdf=False
        )

        val_losses.append(val_loss)
        avg_euc_end_effector_errors.append(avg_euc_end_effector_error)
        
        print(f'Epoch: {epoch} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} R2 {R2_score:.4f}\n'
              f'Average Euclidian End Effector Error: {avg_euc_end_effector_error} '
              f'Max Euclidian End Effector Error:{max_euc_end_effector_error} '
              f'Time per iteration: {1000*avg_iter_time:.4f} ms\n----------')

    best_val_loss = np.min(val_losses)
    best_avg_euc_end_effector_error = np.min(avg_euc_end_effector_errors)
    best_epoch = np.argmin(avg_euc_end_effector_errors)
    
    return best_val_loss, best_avg_euc_end_effector_error, best_epoch

# Define the training function
def train_model(config, my_config,device='cpu', checkpoint_dir=None):


    my_config = update_config(config,my_config)

    # Instantiate the model
    model = get_model(config=my_config)

    optimizer = AdamW(model.parameters(), lr=my_config["learning_rate"], weight_decay=my_config["weight_decay"])
    
    # Load the dataset
    trainDataset = FMGPoseDataset(my_config,mode='train')
    testDataset = FMGPoseDataset(my_config,mode='test',feature_scalar=trainDataset.feature_scalar,label_scalar= trainDataset.label_scalar)

    # Load data
    trainDataloader = DataLoader(trainDataset, batch_size=my_config["batch_size"], shuffle=True,drop_last=True)
    testDataloader = DataLoader(testDataset, batch_size=my_config["batch_size"], shuffle=False,drop_last=True)

    model.to(device)

    
    best_val_loss, best_avg_euc_end_effector_error, best_epoch = train(config,model,optimizer, trainDataloader,testDataloader,trainDataset.label_scalar,
                                                 device,wandb_run=None)


    # Send the current validation loss to Ray Tune
    report(dict(loss=best_val_loss,wrist_error=best_avg_euc_end_effector_error,best_epoch=best_epoch))