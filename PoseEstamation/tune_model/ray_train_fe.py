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
from data_processes.data_loader import FMGPoseDataset_important_sensors
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
    avg_wrist_errors = []
    wrist_stds = []
    avg_elbow_errors = []
    elbow_stds = []

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
        val_loss,avg_critic_loss,avg_iter_time, avg_location_error,avg_wrist_error,max_euc_end_effector_error,R2_score,avg_elbow_error,wrist_std,elbow_std = test_model(
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
        avg_wrist_errors = np.append(avg_wrist_errors,avg_wrist_error)
        wrist_stds.append(wrist_std)
        avg_elbow_errors.append(avg_elbow_error)
        elbow_stds.append(elbow_std)


        
        print(f'Epoch: {epoch} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} R2 {R2_score:.4f}\n'
              f'Average Euclidian End Effector Error: {avg_wrist_errors} '
              f'Max Euclidian End Effector Error:{max_euc_end_effector_error} '
              f'Time per iteration: {1000*avg_iter_time:.4f} ms\n----------')

    
    best_epoch = np.argmin(avg_elbow_errors)
    best_val_loss = val_losses[best_epoch]
    wrist_eror = avg_wrist_errors[best_epoch]
    wrist_std_eror = wrist_stds[best_epoch]
    elbow_eror = avg_elbow_errors[best_epoch]
    elbow_std_eror = elbow_stds[best_epoch]
    
    return best_val_loss, wrist_eror,wrist_std_eror,elbow_eror,elbow_std_eror ,best_epoch


# Define the training function
def train_model(config, my_config,device='cpu', checkpoint_dir=None,score=None):


    my_config = update_config(config,my_config)
    my_config["experiment_name"] = f"ray_best_sensors_{score}_"

    
    feature_impotance_scores_dict ={
        1: 1.39, 2: 4.93, 3: 12.82, 4: 0.99, 5: 3.37, 6: 7.88, 7: 0.5, 8: 5.88,
        9: 9.52, 10: 4.94, 11: 0.89, 12: 1.25, 13: 0.07, 14: 0.01, 15: 0.66, 16: 1.58,
        17: 0.99, 18: 2.36, 19: 1.63, 20: 6.49, 21: 8.23, 22: 1.07, 23: 0.94, 24: 4.65,
        25: 0.62, 26: 1.18, 27: 1.07, 28: 1.16, 29: 1.11, 30: 0.91, 31: 0.49, 32: 13.35
        }
    sensor_mapping = {
        1: 0, 2: 3, 3: 2, 4: 1, 5: 20, 6: 21, 7: 25, 8: 23, 9: 22, 10: 24,
        11: 8, 12: 9, 13: 6, 14: 7, 15: 4, 16: 5, 17: 12, 18: 13, 19: 15,
        20: 14, 21: 31, 22: 11, 23: 10, 24: 28, 25: 29, 26: 30, 27: 16,
        28: 26, 29: 27, 30: 19, 31: 18, 32: 17
    }
    

    mapped_feature_impotance_scores = {}
    for k,v in feature_impotance_scores_dict.items():
        mapped_feature_impotance_scores[k] = feature_impotance_scores_dict[sensor_mapping[k]+ 1]
    
    sensors_to_use = []
    for k,v in mapped_feature_impotance_scores.items():
        if v > score:
            sensors_to_use.append(k)
    
    # filtered_scores = np.array(list(mapped_feature_impotance_scores.values()))

    # filtered_scores = filtered_scores[filtered_scores > score]

    my_config["input_size"] = len(sensors_to_use)
    print("-----------------")
    print(f"Using {len(sensors_to_use)} sensors")
    print("-----------------")

    # Load the dataset
    trainDataset = FMGPoseDataset_important_sensors(my_config,mode='train',sensor_to_use=sensors_to_use)
    testDataset = FMGPoseDataset_important_sensors(my_config,mode='test',sensor_to_use=sensors_to_use,feature_scalar=trainDataset.feature_scalar,label_scalar= trainDataset.label_scalar)

    # Instantiate the model
    model = get_model(config=my_config)

    optimizer = AdamW(model.parameters(), lr=my_config["learning_rate"], weight_decay=my_config["weight_decay"])
    
    # Load data
    trainDataloader = DataLoader(trainDataset, batch_size=my_config["batch_size"], shuffle=True,drop_last=True)
    testDataloader = DataLoader(testDataset, batch_size=my_config["batch_size"], shuffle=False,drop_last=True)

    model.to(device)

    
    best_val_loss, wrist_eror,wrist_std_eror,elbow_eror,elbow_std_eror ,best_epoch = train(my_config,model,optimizer, trainDataloader,testDataloader,trainDataset.label_scalar,
                                                 device,wandb_run=None)


    # Send the current validation loss to Ray Tune
    report(dict(loss=best_val_loss,wrist_error=wrist_eror,wrist_std_eror=wrist_std_eror,elbow_eror=elbow_eror,elbow_std_eror=elbow_std_eror,best_epoch=best_epoch))