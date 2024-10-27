import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
from os import listdir
from os.path import join

import torch.nn.functional as F

import torch
import torch.nn as nn
<<<<<<< HEAD
from torch.optim import Adam,RAdam,AdamW
=======
from torch.optim import Adam,RAdam
>>>>>>> 1276685 (26/08)
from torch.nn import MSELoss
from torch.nn.utils import clip_grad_norm_
from sklearn.preprocessing import MinMaxScaler
from models.models import series_decomp ,VAE
import wandb
import time
from torch.optim.lr_scheduler import LambdaLR , StepLR,CosineAnnealingWarmRestarts 
from transformers import get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup
import random
import torch.optim as optim
import math
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from copy import deepcopy
from scipy.signal import savgol_filter

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))

class ExponentialLoss(nn.Module):
    def __init__(self):
        super(ExponentialLoss, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.mean(torch.exp(torch.abs(y_pred - y_true)))
    
class SinusoidalLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, base_lr, max_lr, step_size_up, last_epoch=-1):
        
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        super(SinusoidalLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        cycle = math.floor(1 + self.last_epoch / (2 * self.step_size_up))
        x = abs(self.last_epoch / self.step_size_up - 2 * cycle + 1)
        lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x)) * math.sin(math.pi * x)

        return [lr for base_lr in self.base_lrs]
class Location_Eror(nn.Module):
    def __init__(self):
        super(Location_Eror, self).__init__()

    def forward(self, predictions, targets):
        # Compute the squared differences
        differences = predictions - targets
        return differences

def TSS(y_true):
    # Total Sum of Squares: Compare actual values to the mean of the actual values
    y_mean = np.mean(y_true)
    return np.sum((y_true - y_mean)**2)

def RSS(y_true, y_pred):
    # Mean Squared Error: Compare actual values to predicted values
    return np.sum((y_true - y_pred)**2)

def R2(tss, rss):
    # R^2 = 1 - (RSS / TSS)
    # mse = MSE(y_true, y_pred)
    # tss = TSS(y_true)
    tss = np.sum(tss)  # Total Sum of Squares
    rss = np.sum(rss)  # RSS = MSE * number of observations
    return 1 - (rss / tss)

def inverse_transform(tensor, mean, std):
    return tensor * std + mean

# Function to convert tensors to numpy arrays and store them
def store_results(outputs, targets, all_outputs, all_targets):
    outputs_np = outputs.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    if all_outputs is None :
        all_outputs = outputs_np
    else:
        all_outputs = np.vstack((all_outputs, outputs_np))
    if all_targets is None :
        all_targets = targets_np
    else:
        all_targets = np.vstack((all_targets, targets_np))
<<<<<<< HEAD
=======

    return all_outputs, all_targets

# Example usage:
# Assuming 'optimizer' is a PyTorch optimizer
# base_lr = 0.01  # Minimum learning rate
# max_lr = 0.1   # Maximum learning rate
# step_size_up = 1000  # Number of training iterations for half a sinusoidal cycle
# scheduler = SinusoidalLR(optimizer, base_lr, max_lr, step_size_up)
# TODO: add time for iteration for the eval model 
def train(config, train_loader, val_loader,model,data_processor, device='cpu',wandb_run=None,critic =None,modelV=None):
    
    # input_size = 9  # Number of features for each time step
    # hidden_size = 128
    # latent_size = 2
    # # Initialize the VAE model
    # vae = VAE(input_size, hidden_size, latent_size).to(device)
    if config.velocity_model:
        optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        optimizerV = Adam(modelV.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # VAE_optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    # optimizer = RAdam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # Check if the model is one of the specified types for custom learning rate scheduling
    if config.use_schedualer:
        if model.name in ["TransformerModel", "DecompTransformerModel", "PatchTST"]:
            warmup_steps = config.warmup_steps
            initial_lr = config.learning_rate  # Starting learning rate
>>>>>>> 1276685 (26/08)

    return all_outputs, all_targets

def init_scheduler(optimizer, config):
    if config.scheduler == 'StepLR':
        scheduler = StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    # elif config.scheduler == 'cosine_decay':
        # scheduler = CosineDecayLR(optimizer, T_0=config.T_0, T_mult=config.T_mult)
    elif config.scheduler == 'NoamLR':
        warmup_steps = config.warmup_steps
        initial_lr = config.learning_rate
        lr_lambda = lambda step: min((step + 1) ** -0.5, (step + 1) * warmup_steps ** -1.5)
        scheduler = LambdaLR(optimizer, lr_lambda)
    # elif config.scheduler == 'PolynomialLR':
    #     scheduler = PolynomialLR(optimizer, max_decay_steps=config.max_decay_steps, end_learning_rate=config.end_learning_rate)
    else:
        scheduler = None
    return scheduler

def train(config, train_loader, val_loader,model,data_processor, device='cpu',wandb_run=None,critic =None,modelV=None):
    
    # input_size = 9  # Number of features for each time step
    # hidden_size = 128
    # latent_size = 2
    # # Initialize the VAE model
    # vae = VAE(input_size, hidden_size, latent_size).to(device)
    if config.velocity_model:
        optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        optimizerV = Adam(modelV.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        # optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    num_training_steps = len(train_loader) * config.num_epochs
    num_warmup_steps = int(0.2 * num_training_steps)  # 10% warmup

    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, 
    #     num_warmup_steps=num_warmup_steps, 
    #     num_training_steps=num_training_steps
    # )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps,
        num_cycles = 0.5
    )

    criterion = RMSELoss() if config.loss_func == 'RMSELoss' else MSELoss()
    criterionV = RMSELoss() if config.loss_func == 'RMSELoss' else MSELoss()
    train_losses = []
    val_losses = []
    TSS_losses = []
    RSS_losses = []
    R_squerd = []
    best_wrist_error = 5000
    best_val_loss = 10
    print("training starts")
    
    # Initialize empty arrays to store the outputs and targets
    all_outputs = None
    all_targets = None
    for epoch in range(config.num_epochs):
        # Initialize the epoch loss and accuracy
        train_loss = 0
        model.train()
        # modelV.train()
        train_loss, train_loss_loc, train_loss_v = 0, 0, 0

        # Wrap your training loop iterator with tqdm
        train_iterator = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")
        # Train on the training set
        for i,(inputs, targets , time_feature) in train_iterator:


            inputs = inputs.to(device=device)
            targets = targets.to(device=device)
            time_feature = time_feature.to(device=device)
            if config.velocity_model:
                # Zero the gradients
                optimizer.zero_grad()
                optimizerV.zero_grad()


                outputs_loc= model(inputs)
                outputs_V = modelV(inputs)

                outputs_loc = outputs_loc[:,-1,:]
                outputs_V = outputs_V[:,-1,:]

                targets_loc = targets[:,-1,:9]
                targets_V = targets[:,-1,9:]

                loss_loc = criterion(outputs_loc, targets_loc)
                loss_v = criterion(outputs_V, targets_V)
                loss = loss_loc + loss_v
                loss.backward()

            else:
                optimizer.zero_grad()

                outputs = model(inputs)
                outputs = outputs[:,-1,:]
                targets = targets[:,-1,:]
                loss = criterion(outputs, targets)
                loss.backward()
                # Store results
                all_outputs, all_targets = store_results(outputs, targets, all_outputs, all_targets)

<<<<<<< HEAD

            train_loss +=  loss.item()
            TSS_losses.append(TSS(targets.cpu().detach().numpy()))
            RSS_losses.append(RSS(targets.cpu().detach().numpy(),outputs.cpu().detach().numpy()))

            optimizer.step()
            if scheduler: scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            train_iterator.set_description(f"Epoch [{epoch}/{config.num_epochs}] Train Loss: {(train_loss/(i+1)):.4f} R^2:{R2(TSS_losses,RSS_losses):.3f} LR: {current_lr:.6f}")
        
        # if scheduler: scheduler.step()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        # # Convert results to a DataFrame
        # df_all_outputs = pd.DataFrame(all_outputs)
        # df_all_targets = pd.DataFrame(all_targets)

        # # Save DataFrame to a CSV file
        # df_all_outputs.to_csv('all_training_outputs.csv', index=False)
        # df_all_targets.to_csv('all_training_targets.csv', index=False)

        val_loss,val_critic_loss,avg_iter_time, avg_location_error,avg_euc_end_effector_error,max_euc_end_effector_error,R2_score = test_model(
=======
            train_loss +=  loss.item()
            optimizer.step()
            current_lr = optimizer.param_groups[0]['lr']
            train_iterator.set_description(f"Epoch [{epoch}/{config.num_epochs}] Train Loss: {(train_loss/(i+1)):.4f} LR: {current_lr:.6f}")
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        # Convert results to a DataFrame
        df_all_outputs = pd.DataFrame(all_outputs)
        df_all_targets = pd.DataFrame(all_targets)

        # Save DataFrame to a CSV file
        df_all_outputs.to_csv('all_training_outputs.csv', index=False)
        df_all_targets.to_csv('all_training_targets.csv', index=False)

        val_loss,val_critic_loss,avg_iter_time, avg_location_error,avg_euc_end_effector_error,max_euc_end_effector_error = test_model(
>>>>>>> 1276685 (26/08)
            model=model,
            critic=critic,
            config=config,
            data_loader=val_loader,
            data_processor=data_processor,
            device=device,
            epoch=epoch,
            task='validate',
            make_pdf= wandb_run,
        )


<<<<<<< HEAD
        print(f'Epoch: {epoch} Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f} R2 {R2_score:.4f} \n Avarege Euclidian End Effector Error: {avg_euc_end_effector_error} Max Euclidian End Effector Error:{max_euc_end_effector_error} time for one iteration {1000*avg_iter_time:.4f} ms \n ----------')
=======
        print(f'Epoch: {epoch} Train Loss: {train_loss}  Val Loss: {val_loss} \n Avarege Euclidian End Effector Error: {avg_euc_end_effector_error} Max Euclidian End Effector Error:{max_euc_end_effector_error} time for one iteration {1000*avg_iter_time:.4f} ms \n ----------')
>>>>>>> 1276685 (26/08)
        # Save the validation loss 
        val_losses.append(val_loss)
        
        if config.wandb_on:
            # log metrics to wandb
            wandb.log({"Train_Loss": train_loss, "Val_loss": val_loss, "Val_Max_Euclidian_End_Effector_Error" : max_euc_end_effector_error , "Val_Avarege_Euclidian_End_Effector_Error": avg_euc_end_effector_error})

        if( avg_euc_end_effector_error < best_wrist_error):
        # if( best_val_loss > val_loss):
            best_wrist_error = avg_euc_end_effector_error
            # best_val_loss = val_loss
            time_stamp = time.strftime("%d_%m_%H_%M", time.gmtime())
            filename = model.name + '_epoch_' +str(epoch)+'_date_'+time_stamp + '.pt'
            best_model_checkpoint_path = join(config.model_path,filename)
            best_model_checkpoint = {
                'epoch': epoch,
                'model_state_dict': deepcopy(model.state_dict()),
                'optimizer_state_dict': deepcopy(optimizer.state_dict()),
                # 'scaler_state_dict': scaler.state_dict(),
                'loss': val_loss,
                'config': config if not config.wandb_on else 'in_wandb' ,
                }
            # Extract the state of the label_scaler
            std_label_scaler_state = {
                'mean': data_processor.label_scaler.mean_.tolist(),
                'var': data_processor.label_scaler.var_.tolist(),
                'scale': data_processor.label_scaler.scale_.tolist(),
                'n_samples_seen': data_processor.label_scaler.n_samples_seen_
            }
                        # Extract the state of the label_scaler
            std_feature_scaler_state = {
                'mean': data_processor.feature_scaler.mean_.tolist(),
                'var': data_processor.feature_scaler.var_.tolist(),
                'scale': data_processor.feature_scaler.scale_.tolist(),
                'n_samples_seen': data_processor.feature_scaler.n_samples_seen_
            }

            # Add the scaler state to your checkpoint dictionary
            best_model_checkpoint['std_label_scaler_state'] = std_label_scaler_state
            best_model_checkpoint['std_feature_scaler_state'] = std_feature_scaler_state
    # filename = 'VAE' +'_date_'+time_stamp + '.pt'
    # vae_model_checkpoint = {
    #     'model_state_dict': deepcopy(vae.state_dict())
    # }
    torch.save(best_model_checkpoint,best_model_checkpoint_path)
    # torch.save(vae_model_checkpoint,filename)
    print(f"model {filename} saved ")
    
    return best_model_checkpoint_path ,best_model_checkpoint

# def train(config, train_loader, val_loader, model, data_processor, device='cpu', wandb_run=None, critic=None, modelV=None):
#     optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
#     if config.velocity_model:
#         optimizerV = Adam(modelV.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
#     criterion = RMSELoss() if config.loss_func == 'RMSELoss' else MSELoss()
    
#     train_losses, val_losses = [], []
#     best_performance = float('inf')

#     for epoch in range(config.num_epochs):
#         model.train()
#         if config.velocity_model:
#             modelV.train()
        
#         train_loss, train_loss_loc, train_loss_v = 0, 0, 0
        
#         train_iterator = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")
#         for i, (inputs, targets, time_feature) in train_iterator:
#             inputs = inputs.to(device)
#             targets = targets.to(device)
#             time_feature = time_feature.to(device)

#             optimizer.zero_grad()
#             if config.velocity_model:
#                 optimizerV.zero_grad()

#             loss = compute_loss(config, model, modelV, inputs, targets, criterion)

#             loss.backward()
#             optimizer.step()
#             if config.velocity_model:
#                 optimizerV.step()

#             train_loss += loss.item()
#             if config.velocity_model:
#                 train_loss_loc += loss.item()  # Adjust this if you're tracking separate losses
#                 train_loss_v += loss.item()    # Adjust this if you're tracking separate losses

#             current_lr = optimizer.param_groups[0]['lr']
#             train_iterator.set_description(f"Epoch [{epoch}/{config.num_epochs}] Train Loss: {(train_loss/(i+1)):.4f} LR: {current_lr:.6f}")

#         train_loss /= len(train_loader)
#         train_losses.append(train_loss)
        
#         val_metrics = test_model(
#             model=model, modelV=modelV if config.velocity_model else None,
#             critic=critic, config=config, data_loader=val_loader,
#             data_processor=data_processor, device=device, epoch=epoch,
#             task='validate', make_pdf=wandb_run
#         )

#         val_loss = val_metrics['val_loss']
#         avg_euc_end_effector_error = val_metrics['avg_euc_end_effector_error']
#         max_euc_end_effector_error = val_metrics['max_euc_end_effector_error']

#         print(f'Epoch: {epoch} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}\n'
#               f'Avg Euclidean End Effector Error: {avg_euc_end_effector_error:.4f} '
#               f'Max Euclidean End Effector Error: {max_euc_end_effector_error:.4f}\n----------')

#         val_losses.append(val_loss)
        
#         if config.wandb_on:
#             log_data = {
#                 "Train_Loss": train_loss,
#                 "Val_loss": val_loss,
#                 "Val_Max_Euclidian_End_Effector_Error": max_euc_end_effector_error,
#                 "Val_Average_Euclidian_End_Effector_Error": avg_euc_end_effector_error
#             }
#             if config.velocity_model:
#                 log_data.update({
#                     "Train_Loss_Loc": train_loss_loc / len(train_loader),
#                     "Train_Loss_Vel": train_loss_v / len(train_loader)
#                 })
#             wandb.log(log_data)

#         if avg_euc_end_effector_error < best_performance:
#             best_performance = avg_euc_end_effector_error
#             best_model_checkpoint_path, best_model_checkpoint = save_checkpoint(
#                 config, epoch, model, modelV, optimizer, optimizerV, val_loss, data_processor
#             )

#     return best_model_checkpoint_path, best_model_checkpoint

def compute_loss(config, model, modelV, inputs, targets, criterion):
    if config.velocity_model:
        outputs_loc = model(inputs)[:, -1, :]
        outputs_V = modelV(inputs)[:, -1, :]
        targets_loc = targets[:, -1, :9]
        targets_V = targets[:, -1, 9:]
        loss_loc = criterion(outputs_loc, targets_loc)
        loss_v = criterion(outputs_V, targets_V)
        return config.loc_loss_weight * loss_loc + config.vel_loss_weight * loss_v
    else:
        outputs = model(inputs)[:, -1, :]
        targets = targets[:, -1, :]
        return criterion(outputs, targets)

def save_checkpoint(config, epoch, model, modelV, optimizer, optimizerV, val_loss, data_processor):
    time_stamp = time.strftime("%d_%m_%H_%M", time.gmtime())
    filename = f'{model.name}_epoch_{epoch}_date_{time_stamp}.pt'
    checkpoint_path = join(config.model_path, filename)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': deepcopy(model.state_dict()),
        'optimizer_state_dict': deepcopy(optimizer.state_dict()),
        'loss': val_loss,
        'config': config if not config.wandb_on else 'in_wandb',
        'std_label_scaler_state': get_scaler_state(data_processor.label_scaler),
        'std_feature_scaler_state': get_scaler_state(data_processor.feature_scaler),
    }
    
    if config.velocity_model:
        checkpoint.update({
            'modelV_state_dict': deepcopy(modelV.state_dict()),
            'optimizerV_state_dict': deepcopy(optimizerV.state_dict()),
        })
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Model {filename} saved")
    
    return checkpoint_path, checkpoint

def get_scaler_state(scaler):
    return {
        'mean': scaler.mean_.tolist(),
        'var': scaler.var_.tolist(),
        'scale': scaler.scale_.tolist(),
        'n_samples_seen': scaler.n_samples_seen_
    }



def compute_loss(config, model, modelV, inputs, targets, criterion):
    if config.velocity_model:
        outputs_loc = model(inputs)[:, -1, :]
        outputs_V = modelV(inputs)[:, -1, :]
        targets_loc = targets[:, -1, :9]
        targets_V = targets[:, -1, 9:]
        loss_loc = criterion(outputs_loc, targets_loc)
        loss_v = criterion(outputs_V, targets_V)
        return config.loc_loss_weight * loss_loc + config.vel_loss_weight * loss_v
    else:
        outputs = model(inputs)[:, -1, :]
        targets = targets[:, -1, :]
        return criterion(outputs, targets)

def save_checkpoint(config, epoch, model, modelV, optimizer, optimizerV, val_loss, data_processor):
    time_stamp = time.strftime("%d_%m_%H_%M", time.gmtime())
    filename = f'{model.name}_epoch_{epoch}_date_{time_stamp}.pt'
    checkpoint_path = join(config.model_path, filename)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': deepcopy(model.state_dict()),
        'optimizer_state_dict': deepcopy(optimizer.state_dict()),
        'loss': val_loss,
        'config': config if not config.wandb_on else 'in_wandb',
        'std_label_scaler_state': get_scaler_state(data_processor.label_scaler),
        'std_feature_scaler_state': get_scaler_state(data_processor.feature_scaler),
    }
    
    if config.velocity_model:
        checkpoint.update({
            'modelV_state_dict': deepcopy(modelV.state_dict()),
            'optimizerV_state_dict': deepcopy(optimizerV.state_dict()),
        })
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Model {filename} saved")
    
    return checkpoint_path, checkpoint

def get_scaler_state(scaler):
    return {
        'mean': scaler.mean_.tolist(),
        'var': scaler.var_.tolist(),
        'scale': scaler.scale_.tolist(),
        'n_samples_seen': scaler.n_samples_seen_
    }


def test_model(model, config ,
                # criterion ,
                data_loader, 
                data_processor ,
                device='cpu',
                make_pdf=False,
                epoch = 0,
                task = 'train',
                critic = None,
                vae = None):

        if config.loss_func == 'MSELoss':
            criterion = MSELoss()
        elif config.loss_func == 'RMSELoss':
            criterion = RMSELoss()
        total_loss = 0
        total_critic_loss = 0
        avg_critic_loss = 0
        total_time = 0
        sum_location_error = 0
        max_euc_end_effector_error = 0
        TSS_losses = []
        RSS_losses = []
        # Evaluate on the validation set
        with torch.no_grad():
            model.eval()
            # Wrap your training loop iterator with tqdm
            train_iterator = tqdm(enumerate(data_loader), total=len(data_loader), desc="Test")
            for i,(inputs, targets , time_feature) in train_iterator:

                inputs = inputs.to(device=device)
                targets = targets.to(device=device)
                time_feature = time_feature.cpu().detach().numpy()
                
                start_time = time.time()

                if config.velocity_model:
                    outputs_loc, outputs_V = model(inputs)
                    outputs_loc = outputs_loc[:,-1,:]
                    outputs_V = outputs_V[:,-1,:]
<<<<<<< HEAD
                    targets_loc = targets[:,-1,:9]
                    targets_V = targets[:,-1,9:]
                    loss_loc = criterion(outputs_loc, targets_loc)
                    loss_v = criterion(outputs_V, targets_V)
                    loss = loss_loc + loss_v
=======
                else:
                    outputs = model(inputs)
                    outputs = outputs[:,-1,:]
                
                if config.velocity_model:
                    targets_loc = targets[:,-1,:9]
                    targets_V = targets[:,-1,9:]
                else:
                    targets = targets[:,-1,:]

                if config.velocity_model:
                    loss_loc = criterion(outputs_loc, targets_loc)
                    loss_v = criterion(outputs_V, targets_V)
                    loss = loss_loc + loss_v
                else:
                    loss = criterion(outputs, targets)
                
                # outputs,_,_ = vae(outputs)
                # outputs = outputs[:,-1,:]
                # targets = targets[:,-1,:]
                if config.velocity_model:

>>>>>>> 1276685 (26/08)
                    outputs_loc = outputs_loc.cpu().detach().numpy()
                    targets_loc = targets_loc.cpu().detach().numpy()
                    outputs_V = outputs_V.cpu().detach().numpy()
                    targets_V = targets_V.cpu().detach().numpy()
                    
                    outputs = np.concatenate([outputs_loc,outputs_V],axis=1)
                    targets = np.concatenate([targets_loc,targets_V],axis=1)
<<<<<<< HEAD
                else:
                    outputs = model(inputs)
                    outputs = outputs[:,-1,:]
                    targets = targets[:,-1,:]
                    loss = criterion(outputs, targets)
                    outputs = outputs.cpu().detach().numpy()
                    targets = targets.cpu().detach().numpy()
                    TSS_losses.append(TSS(targets))
                    RSS_losses.append(RSS(targets,outputs))
=======

                else:
                    outputs = outputs.cpu().detach().numpy()
                    targets = targets.cpu().detach().numpy()
>>>>>>> 1276685 (26/08)


                if config.norm_labels:
                    unnorm_outputs = data_processor.label_scaler.inverse_transform(outputs)
                    unnorm_targets = data_processor.label_scaler.inverse_transform(targets)

                else :
                    unnorm_outputs = outputs
                    unnorm_targets = targets
                
<<<<<<< HEAD
=======


>>>>>>> 1276685 (26/08)
                location_error = (np.abs(unnorm_outputs - unnorm_targets).sum(axis=0))/inputs.size(0)
                
                if i == 0:
                    predsToPlot = unnorm_outputs
                    targetsToPlot = unnorm_targets
                    time_featureToPlot = time_feature
                    inputsToSave = inputs.cpu().detach().numpy()

                predsToPlot = np.concatenate((predsToPlot,unnorm_outputs))
                targetsToPlot = np.concatenate((targetsToPlot,unnorm_targets))
                time_featureToPlot = np.concatenate((time_featureToPlot,time_feature))
                inputsToSave = np.concatenate((inputsToSave,inputs.cpu().detach().numpy()))
                
                

                end_time = time.time()
                total_time += (end_time - start_time)
                total_loss += loss.item()
                sum_location_error += location_error

            # # Apply Savitzky-Golay Filter
            window_size = 100
            poly_order = 5
<<<<<<< HEAD
            # predsToPlot = savgol_filter(predsToPlot,deriv=0, window_length=window_size, polyorder=poly_order,axis=0)
            # firstOrder_derivative_predsToPlot = savgol_filter(predsToPlot,deriv=1,delta=30.0, window_length=window_size, polyorder=poly_order,axis=0)
=======
            predsToPlot = savgol_filter(predsToPlot,deriv=0, window_length=window_size, polyorder=poly_order,axis=0)
            firstOrder_derivative_predsToPlot = savgol_filter(predsToPlot,deriv=1,delta=30.0, window_length=window_size, polyorder=poly_order,axis=0)
>>>>>>> 1276685 (26/08)
            # predsToPlot += firstOrder_derivative_predsToPlot*200 

            # Reshape the arrays to separate the coordinates of each point
            if config.velocity_model:
                true_reshaped = targetsToPlot.reshape(-1, 6, 3)  # shape will be [n, 3, 3]
                pred_reshaped = predsToPlot.reshape(-1, 6, 3)  # shape will be [n, 3, 3]
            else:
                true_reshaped = targetsToPlot.reshape(-1, 3, 3)  # shape will be [n, 3, 3]
                pred_reshaped = predsToPlot.reshape(-1, 3, 3)  # shape will be [n, 3, 3]
            # Calculate the Euclidean distance for each corresponding point
            distances = np.linalg.norm(true_reshaped - pred_reshaped, axis=2)

            # Calculate the mean error
            mean_error = np.mean(distances,axis=0)

            avg_loss = total_loss/len(data_loader)
            avg_iter_time = total_time/len(data_loader)
            
            avg_location_error = sum_location_error/len(data_loader)


            title = (f"Model: {model.name}, Task: {task}\n "
                    f"RMSE Avg Loss: {avg_loss}\n "
                    f"RMSE Avg Critic Loss: {avg_critic_loss}\n "
                    f"Avg Iter Time: {avg_iter_time}\n "
                    # f"Avg Location Error: {avg_location_error}\n "
                    f"Avg Euclidean wrist Error: {mean_error[-1]}\n"
                    f"Avg Euclidean Elbow Error: {mean_error[1]}\n"
                    f"Avg Euclidean Sholder Error: {mean_error[0]}\n"
                    f"Max Euclidean End Effector Error: {distances.max()}\n")
            
            # to save df 
            # np.save('inputs.npy',inputsToSave)

            # df_inputs = pd.DataFrame(inputsToSave,columns=config.fmg_index)
            # df_true = pd.DataFrame(targetsToPlot,columns=config.label_index)
            # df_preds = pd.DataFrame(predsToPlot,columns=config.label_index)
            # df_true.to_csv('df_true.csv')
            # df_preds.to_csv('df_preds.csv')
            # df_inputs.to_csv('df_inputs.csv')

            if not make_pdf:
                if config.num_labels >3:
                    if config.velocity_model:
                        legends = config.label_index + config.velocity_label_inedx # ['MEx','MEY' ...]
                        for i in range(12):
                            plt.subplot(12,1,(i+1))
                            idx = i
                            if i < 6:
                                add = 3
                            else:
                                add = 6
                            plt.plot(targetsToPlot[:,idx+add:idx+add+1], linestyle='-')
                            plt.legend([legends[i+add]])
                            plt.plot(predsToPlot[:,idx+add:idx+add+1], linestyle='--')
                    else:
                        legends = config.label_index # ['MEx','MEY' ...]
                        for i in range(6):
                            plt.subplot(6,1,(i+1))
                            idx = i
                            plt.plot(targetsToPlot[:,idx+3:idx+3+1], linestyle='-')
                            plt.legend([legends[i+3]])
                            plt.plot(predsToPlot[:,idx+3:idx+3+1], linestyle='--')
                            # Set the x-axis limits
                            # plt.ylim(-0.3, 0.7)

                else:
                    plt.plot(targetsToPlot, linestyle='-')
                    plt.plot(predsToPlot, linestyle='--')

                # plt.suptitle(title)


                dir_path = './results/' + model.name

                # Check if the directory exists
                if not os.path.exists(dir_path):
                    # Create the directory if it does not exist
                    os.makedirs(dir_path)
                if task == "test":
                    name =  model.name +'test' + '.pdf'
                    name_txt = model.name +'test' + '.txt'
                else:
                    name =  model.name +'_epoch_'+ str(epoch)  + '.pdf'
                    name_txt = model.name +'_epoch_'+ str(epoch)  + '.txt'
                with open(os.path.join(dir_path,name_txt),'w') as f :
                    f.write(title)
                    f.write(str(config))
                plt.savefig(os.path.join(dir_path,name))
                plt.close()
            
        avg_euc_end_effector_error = mean_error[-1]
        max_euc_end_effector_error = distances.max()
        R2_score = R2(TSS_losses,RSS_losses)
        return avg_loss,avg_critic_loss,avg_iter_time, avg_location_error,avg_euc_end_effector_error,max_euc_end_effector_error,R2_score


<<<<<<< HEAD
=======
# def test_model(model, config, data_loader, data_processor, device='cpu', make_pdf=False, epoch=0, task='train', critic=None, modelV=None):
#     criterion = MSELoss() if config.loss_func == 'MSELoss' else RMSELoss()
#     total_loss = 0
#     total_time = 0
#     sum_location_error = 0

#     with torch.no_grad():
#         model.eval()
#         if config.velocity_model:
#             modelV.eval()
        
#         test_iterator = tqdm(enumerate(data_loader), total=len(data_loader), desc="Test")
#         for i, (inputs, targets, time_feature) in test_iterator:
#             inputs = inputs.to(device=device)
#             targets = targets.to(device=device)
#             time_feature = time_feature.cpu().detach().numpy()
            
#             start_time = time.time()

#             if config.velocity_model:
#                 outputs_loc = model(inputs)[:, -1, :]
#                 outputs_V = modelV(inputs)[:, -1, :]
#                 targets_loc = targets[:, -1, :9]
#                 targets_V = targets[:, -1, 9:]
#                 loss_loc = criterion(outputs_loc, targets_loc)
#                 loss_v = criterion(outputs_V, targets_V)
#                 loss = loss_loc + loss_v
#                 outputs_loc = outputs_loc.cpu().detach().numpy()
#                 outputs_V = outputs_V.cpu().detach().numpy()
#                 targets_loc = targets_loc.cpu().detach().numpy()
#                 targets_V = targets_V.cpu().detach().numpy()
#                 outputs = np.concatenate([outputs_loc, outputs_V], axis=1)
#                 targets = np.concatenate([targets_loc, targets_V], axis=1)
#             else:
#                 outputs = model(inputs)[:, -1, :]
#                 targets = targets[:, -1, :]
#                 loss = criterion(outputs, targets)
#                 outputs = outputs.cpu().detach().numpy()
#                 targets = targets.cpu().detach().numpy()

#             if config.norm_labels:
#                 unnorm_outputs = data_processor.label_scaler.inverse_transform(outputs)
#                 unnorm_targets = data_processor.label_scaler.inverse_transform(targets)
#             else:
#                 unnorm_outputs = outputs
#                 unnorm_targets = targets

#             location_error = (np.abs(unnorm_outputs - unnorm_targets).sum(axis=0)) / inputs.size(0)
            
#             if i == 0:
#                 predsToPlot = unnorm_outputs
#                 targetsToPlot = unnorm_targets
#                 time_featureToPlot = time_feature
#                 inputsToSave = inputs.cpu().detach().numpy()
#             else:
#                 predsToPlot = np.concatenate((predsToPlot, unnorm_outputs))
#                 targetsToPlot = np.concatenate((targetsToPlot, unnorm_targets))
#                 time_featureToPlot = np.concatenate((time_featureToPlot, time_feature))
#                 inputsToSave = np.concatenate((inputsToSave, inputs.cpu().detach().numpy()))

#             end_time = time.time()
#             total_time += (end_time - start_time)
#             total_loss += loss.item()
#             sum_location_error += location_error

#         # Apply Savitzky-Golay Filter
#         window_size = 300
#         poly_order = 5
#         predsToPlot = savgol_filter(predsToPlot, window_length=window_size, polyorder=poly_order, axis=0)

#         # Reshape the arrays to separate the coordinates of each point
#         true_reshaped = targetsToPlot.reshape(-1, 6 if config.velocity_model else 3, 3)
#         pred_reshaped = predsToPlot.reshape(-1, 6 if config.velocity_model else 3, 3)

#         # Calculate the Euclidean distance for each corresponding point
#         distances = np.linalg.norm(true_reshaped - pred_reshaped, axis=2)

#         # Calculate the mean error
#         mean_error = np.mean(distances, axis=0)

#         avg_loss = total_loss / len(data_loader)
#         avg_iter_time = total_time / len(data_loader)
#         avg_location_error = sum_location_error / len(data_loader)

#         title = (f"Model: {model.name}, Task: {task}\n "
#                 f"RMSE Avg Loss: {avg_loss}\n "
#                 f"Avg Iter Time: {avg_iter_time}\n "
#                 f"Avg Euclidean wrist Error: {mean_error[-1]}\n"
#                 f"Avg Euclidean Elbow Error: {mean_error[1]}\n"
#                 f"Avg Euclidean Shoulder Error: {mean_error[0]}\n"
#                 f"Max Euclidean End Effector Error: {distances.max()}\n")

#         if not make_pdf:
#             plot_results(config, targetsToPlot, predsToPlot)
#             save_results(config, model, task, epoch, title, dir_path='./results/')

#     avg_euc_end_effector_error = mean_error[-1]
#     max_euc_end_effector_error = distances.max()
    
#     return {
#         'val_loss': avg_loss,
#         'avg_iter_time': avg_iter_time,
#         'avg_location_error': avg_location_error,
#         'avg_euc_end_effector_error': avg_euc_end_effector_error,
#         'max_euc_end_effector_error': max_euc_end_effector_error
#     }

>>>>>>> 1276685 (26/08)
def plot_results(config, targetsToPlot, predsToPlot):
    if config.num_labels > 3:
        if config.velocity_model:
            legends = config.label_index + config.velocity_label_index
            for i in range(12):
                plt.subplot(12, 1, (i+1))
                idx = i
                add = 3 if i < 6 else 6
                plt.plot(targetsToPlot[:, idx+add:idx+add+1], linestyle='-')
                plt.legend([legends[i+add]])
                plt.plot(predsToPlot[:, idx+add:idx+add+1], linestyle='--')
        else:
            legends = config.label_index
            for i in range(6):
                plt.subplot(6, 1, (i+1))
                idx = i
                plt.plot(targetsToPlot[:, idx+3:idx+3+1], linestyle='-')
                plt.legend([legends[i+3]])
                plt.plot(predsToPlot[:, idx+3:idx+3+1], linestyle='--')
    else:
        plt.plot(targetsToPlot, linestyle='-')
        plt.plot(predsToPlot, linestyle='--')

def save_results(config, model, task, epoch, title, dir_path='./results/'):
    model_dir = os.path.join(dir_path, model.name)
    os.makedirs(model_dir, exist_ok=True)
    
    if task == "test":
        name = f"{model.name}test"
    else:
        name = f"{model.name}_epoch_{epoch}"
    
    with open(os.path.join(model_dir, f"{name}.txt"), 'w') as f:
        f.write(title)
        f.write(str(config))
    
    plt.savefig(os.path.join(model_dir, f"{name}.pdf"))
    plt.close()


def calculate_ema(values, span):
    alpha = 2 / (span + 1)
    ema = np.zeros_like(values)
    ema[0] = values[0]  # Initialize the first EMA value to the first data point

    for i in range(1, len(values)):
        ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]

    return ema
def calculate_ewma_2d(data, alpha):
    """
    Calculate the EWMA for each sequence in a 2D array where each row is a sequence.

    Parameters:
    - data (np.array): 2D array of shape (sequence_length, num_features)
    - alpha (float): Smoothing factor in the range (0, 1)

    Returns:
    - np.array: EWMA of the same shape as data
    """
    # alpha = 2 / (span + 1)
    ewma = np.zeros_like(data)
    ewma[0, :] = data[0, :]  # Initialize the first EWMA value to the first data point

    for i in range(1, data.shape[0]):
        ewma[i, :] = alpha * data[i, :] + (1 - alpha) * ewma[i - 1, :]

    return ewma
# Function to calculate the model size
def print_model_size(model):
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Size: {model_size} parameters")


def plot_losses(config,train_losses, val_losses=[],train=True):
    
    if train:
        # Create a figure and axis
        fig, ax = plt.subplots()
        
        # Plot the training and validation losses
        ax.plot(train_losses, label='Training Loss')
        ax.plot(val_losses, label='Validation Loss')
        
        
        # Add labels and a legend
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
    if config.wandb_on:
        # Log the figure to wandb
        wandb.log({"preds and targets": plt})
    else:
        # Show the plots
        plt.show()
        # Show the plot
        plt.show()

def plot_results(config,data_loader,model,device,data_processor,critic ):

    with torch.no_grad():
        model.eval()
        for i, (inputs, targets , time_feature) in enumerate(data_loader):

            inputs = inputs.to(device=device)
            targets = targets.to(device=device)
            # time_feature = time_feature.to(device=device)

            
            # outputs = model(inputs)
            
            # targets = targets[:,-1,:]
            # outputs = outputs[:,-1,:]

            # if config.norm_labels:
                # unnorm_outputs = data_processor.label_scaler.inverse_transform(outputs.cpu().detach().numpy())
            #     unnorm_targets = data_processor.label_scaler.inverse_transform(targets.cpu().detach().numpy())
            # else:
            #     unnorm_outputs = outputs.cpu().detach().numpy()
            #     unnorm_targets = targets.cpu().detach().numpy()
            # if i == 0:
            #     predsToPlot = unnorm_outputs
            #     targetsToPlot = unnorm_targets
            outputs = model(inputs)
            if config.with_critic:
                critic_out = critic(outputs)

            targets = targets[:,-1,:]
            outputs = outputs[:,-1,:]
            if config.with_critic:
                critic_out = critic_out[:,-1,:]
            time_feature = time_feature[:,-1,:]

            if config.with_critic:
                outputs = outputs - critic_out

            outputs = outputs.cpu().detach().numpy()
            targets = targets.cpu().detach().numpy()

            
            if config.norm_labels:
                unnorm_outputs = data_processor.label_scaler.inverse_transform(outputs)
                unnorm_targets = data_processor.label_scaler.inverse_transform(targets)
            else :
                unnorm_outputs = outputs
                unnorm_targets = targets
            # critic model
            # if config.norm_labels:
            #     target_mean = torch.tensor(data_processor.label_scaler.mean_.tolist(),dtype=torch.float32).to(device)
            #     target_var = torch.tensor(data_processor.label_scaler.var_.tolist(),dtype=torch.float32).to(device)
            #     unnorm_outputs = inverse_transform(outputs,mean=target_mean,std=target_var)
            #     unnorm_targets = inverse_transform(targets,mean=target_mean,std=target_var)
            # else :
            #     unnorm_outputs = outputs
            #     unnorm_targets = targets


            # unnorm_outputs = unnorm_outputs - critic_out
            # if config.norm_labels:
            #     unnorm_outputs = data_processor.label_scaler.inverse_transform(outputs)
            #     unnorm_targets = data_processor.label_scaler.inverse_transform(targets)
            # else :
            #     unnorm_outputs = outputs
            #     unnorm_targets = targets
            if i == 0:
                predsToPlot = unnorm_outputs
                targetsToPlot = unnorm_targets
            predsToPlot = np.concatenate((predsToPlot,unnorm_outputs))
            targetsToPlot = np.concatenate((targetsToPlot,unnorm_targets))  

    # Create a figure and a grid of subplots with 1 row and 2 columns
    fig, axes = plt.subplots(2, 1, figsize=(10, 4),sharex=True,sharey=True)  # Adjust figsize as needed
    # rand_plot = random.randint(0,1)
    # plot_length = 10000
    # start_plot = rand_plot*plot_length
    # end_plot = rand_plot*plot_length +plot_length
    start_plot = 0
    end_plot = 20000
    # Plot data on the first subplot
    axes[0].plot(predsToPlot[:,:12])
    axes[0].set_title('Plot of preds location')
    axes[0].grid()

    if config.with_velocity:
        axes[0,1].plot(predsToPlot[:,12:])
        axes[0,1].set_title('Plot of preds V ')
        axes[0,1].grid()

    # Plot data on the second subplot
    axes[1].plot(targetsToPlot[:,:12])
    axes[1].set_title('Plot of targets location')
    axes[1].grid()

    if config.with_velocity:
        axes[1,1].plot(targetsToPlot[:,12:])
        axes[1,1].set_title('Plot of targets V')
        axes[1,1].grid()
        

    # Adjust layout to prevent overlap
    plt.tight_layout()


    if config.wandb_on:
        # Log the figure to wandb
        wandb.log({"preds and targets": plt})
    else:
        # Show the plots
        plt.show()

def model_eval_metric(config,model,test_loader,
                    data_processor,
                    device='cpu',wandb_run=None):
    # show distance between ground truth and prediction by 3d points [0,M2,M3,M4] 

    model = model.to(device=device)
    model.eval()
    # Evaluate on the test set
    with torch.no_grad():

        inputs = test_loader.dataset.tensors[0]
        targets = test_loader.dataset.tensors[1]
        
        inputs = inputs.to(device=device)
        targets = targets.to(device=device)

        outputs = model(inputs)
        if outputs.dim() == 3:
            outputs = outputs[:,-1,:]
        size = outputs.size(0)
        
        if config.norm_labels:
            outputs = data_processor.label_scaler.inverse_transform(outputs.cpu().detach().numpy())
            targets = data_processor.label_scaler.inverse_transform(targets[:,-1:,:].squeeze(1).cpu().detach().numpy())

        if config.plot_pred:
            plot_results(config,outputs,targets,wandb_run=wandb_run)

        dist = np.sqrt(((outputs - targets)**2).sum(axis=0)/size)

    return dist
def set_seed(seed, torch_deterministic=False, rank=0):
    """ set seed across modules """
    if seed == -1 and torch_deterministic:
        seed = 42 + rank
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    else:
        seed = seed + rank

    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # added these to test if it helps with reproducibility
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed
# depricated
def min_max_unnormalize(data, min_val, max_val,bottom=-1, top=1):

    return torch.tensor(((data-bottom)/(top-bottom)) * (max_val - min_val) + min_val)

#depricated 
def min_max_normalize(data,bottom=-1, top=1):

    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)

    norm = bottom+(data - min_val) / (max_val - min_val)*(top-bottom)

    return norm,max_val,min_val

def plot_data(config,data):
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(40,5))

    ax1.plot(data.drop(['session_time_stamp'],axis=1)[config.positoin_label_inedx])
    ax2.plot(data.drop(['session_time_stamp'],axis=1)[config.fmg_index])
    ax1.legend()

    plt.show() 

# clould be deleted 
def rollig_window(config,data):

    data_avg =data.copy()
    data_avg[config.fmg_index] = data_avg[config.fmg_index].rolling(window=config.window_size, axis=0).mean()

    return data_avg 

def find_bias(df):
    """
    Given a pandas dataframe containing FMG data, finds the bias for each time stamped session and returns a pandas dataframe.
    """
    bias_df = pd.DataFrame()

    for time_stamp in df['session_time_stamp'].unique():

        temp_df = pd.DataFrame(df[df['session_time_stamp'] == time_stamp].drop('session_time_stamp',axis=1),dtype=float).mean().to_frame().T
        temp_df['session_time_stamp'] = time_stamp
        bias_df = pd.concat([bias_df,temp_df],axis= 0,ignore_index=False)

    return bias_df

def find_std(df):
    """
    Given a pandas dataframe containing FMG or IMU data, finds the standard deviation for each time stamped session and returns a pandas dataframe.
    """
    std_df = pd.DataFrame()

    for time_stamp in df['session_time_stamp'].unique():

        temp_df = df[df['session_time_stamp'] == time_stamp].drop('session_time_stamp',axis=1).std().T.copy()
        temp_df['session_time_stamp'] = time_stamp
        std_df = pd.concat([std_df,temp_df],axis=1,ignore_index=False)

    return std_df.T

def subtract_bias(df):
    # Compute the bias for each unique value of the session_time_stamp column
    bias_df = find_bias(df)
    
    # Initialize an empty DataFrame to store the result
    new_df = pd.DataFrame()
    
    # Iterate over each unique value of the session_time_stamp column
    for time_stamp in df['session_time_stamp'].unique():
        # Select the rows of df and bias_df corresponding to the current time stamp
        df_rows = df[df['session_time_stamp'] == time_stamp].copy()
        bias_rows = bias_df[bias_df['session_time_stamp'] == time_stamp].copy()
        
        df_rows= df_rows.drop('session_time_stamp', axis=1).astype(float).copy()
        bias_rows = bias_rows.drop('session_time_stamp', axis=1).astype(float).copy()
        
        # Subtract the bias from the data in df
        temp_df = df_rows-bias_rows.to_numpy() 
        

        # # Add back the session_time_stamp column
        # temp_df['session_time_stamp'] = time_stamp
        
        # Append the result to new_df
        new_df = pd.concat([new_df, temp_df], axis=0, ignore_index=False)
    
    return new_df

def get_label_axis(labels,config):
    #label_inedx = ['M1x','M1y','M1z','M2x','M2y','M2z','M3x','M3y','M3z','M4x','M4y','M4z']
    # Create a copy of the labels DataFrame slice
    # labels_copy = labels[config.first_positoin_label_inedx].copy()
    labels = labels.copy()
    # Now perform the operations on the copy
    labels.loc[:,['M1x','M2x','M3x','M4x']]  = labels[['M1x','M2x','M3x','M4x']].sub(labels['M1x'], axis=0)
    labels.loc[:,['M1y','M2y','M3y','M4y']] = labels[['M1y','M2y','M3y','M4y']].sub(labels['M1y'], axis=0)
    labels.loc[:,['M1z','M2z','M3z','M4z']] = labels[['M1z','M2z','M3z','M4z']].sub(labels['M1z'], axis=0)
   
    return labels[config.position_label_index]

def calc_velocity(config, label_df):
    # Copy the dataframe to avoid SettingWithCopyWarning
    label_df = label_df.copy()
    
    # Time interval in seconds, based on 60 Hz frequency
    delta_t = 1 / config.sample_speed  
    
    # Columns corresponding to positions
    position_cols = config.label_index
    
    # Columns corresponding to velocities
    velocity_cols = config.velocity_label_inedx

    
    # Calculate velocity
    velocity_df = label_df[position_cols].diff() / delta_t
    velocity_df.columns = velocity_cols
    # Update the original dataframe with the calculated velocities
    label_df[velocity_cols] = velocity_df
    return label_df

def mask(data,config):

    # create a mask that selects rows where the values in fmg_index columns are greater than 1024
    mask1 = (data[config.fmg_index] > 1024).any(axis=1)

    # create a mask that selects rows where the values in first_position_label_index columns are greater than 2
    mask2 = (data[config.first_position_label_index] > 3).any(axis=1)

    # combine the masks using the | (or) operator
    mask = mask1 | mask2

    # drop the rows from the DataFrame
    data = data.drop(data[mask].index)


    return data

def is_not_numeric(x):
    try:
        float(x)
        return False
    except ValueError:
        return True
    

def print_not_numeric_vals(df):

    mask = df.drop(['session_time_stamp'],axis=1).applymap(is_not_numeric)
    non_numeric_values = df[mask].stack().dropna()
    print(non_numeric_values)

    return non_numeric_values

# def create_sliding_sequences(input_tensor, sequence_length):

#     sample_size, features = input_tensor.shape
#     new_sample_size = sample_size - sequence_length + 1


#     sequences = []

#     for i in range(new_sample_size):

#         sequence = input_tensor[i:i+sequence_length]
#         sequences.append(sequence)

#     return torch.stack(sequences)

def create_sliding_sequences_old(input_array, sequence_length):
    sample_size, features = input_array.shape
    new_sample_size = sample_size - sequence_length + 1

    sequences = []

    for i in range(new_sample_size):
        sequence = input_array[i:i+sequence_length]
        sequences.append(sequence)

    return np.array(sequences)

def create_sliding_sequences(input_array, sequence_length):
    
    sample_size, features = input_array.shape
    sequences = []
    sampled_array = []
<<<<<<< HEAD
    step_size = 32
=======
    step_size = 1
>>>>>>> 1276685 (26/08)
    i = 0
    jump = 2
    j = 0
    while j <  sample_size - 1:
        sampled_array.append(input_array[j])
        j += jump
    
    sampled_array = np.array(sampled_array)
    sample_size, features = sampled_array.shape
    new_sample_size = sample_size - sequence_length + 1
    while step_size*i + sequence_length <  new_sample_size:
        sequence = sampled_array[i*step_size:i*step_size+sequence_length]
        sequences.append(sequence)
        i+=1

    return np.array(sequences)

def euclidian_end_effector_error(eval_metric):

    return np.sqrt((eval_metric**2).sum()) 

def center_diff(config, locations: pd.DataFrame, h=0.01, order=1) -> pd.DataFrame:
    coefficients = {
    '1':[-0.5,0,0.5],
    '2':[1/12,-2/3,0,2/3,-1/12],
    '3':[-1/60,3/20,-3/4,0,3/4,-3/20,1/60],
    '4':[1/280,-4/105,1/5,-4/5,0,4/5,-1/5,4/105,-1/280],
    }
    vals_index =config.dfmg_index + config.velocity_label_inedx
    locations_index = config.fmg_index + config.label_index 

    locations_with_velocity = locations.copy()
    
    # Prepare a DataFrame to hold the velocity calculations
    velocity_df = pd.DataFrame(index=locations.index, columns=vals_index)
    velocity_df = velocity_df.fillna(0)  # Fill with zeros
    
    coeff = coefficients[str(order)]
    
    # Function to apply the central difference using the coefficients
    def apply_central_diff(row_index):
        if row_index < order or row_index >= len(locations) - order:
            return pd.Series(0, index=vals_index)
        else:
            velocities = []
            for loc in locations_index:
                velocity = np.dot(coeff, locations[loc][row_index-order:row_index+order+1].to_numpy()) / h
                velocities.append(velocity)
            return pd.Series(velocities, index=vals_index)
    
    # Apply the central difference calculation for each valid row
    velocity_df = velocity_df.apply(lambda x: apply_central_diff(x.name), axis=1)
    
    # Concatenate the original locations DataFrame with the velocity DataFrame
    locations_with_velocity = pd.concat([locations_with_velocity, velocity_df], axis=1)
    
    return locations_with_velocity


def center_diff_torch(config, locations: pd.DataFrame, h=0.01, order=1) -> pd.DataFrame:
    coefficients = {
        '1': [-0.5, 0, 0.5],
        '2': [1/12, -2/3, 0, 2/3, -1/12],
        '3': [-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60],
        '4': [1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280],
    }
    
    vals_index = config.velocity_label_inedx
    locations_index = config.label_index
    
    locations_with_velocity = locations.copy()
    
    # Prepare a DataFrame to hold the velocity calculations
    velocity_df = pd.DataFrame(index=locations.index, columns=vals_index)
    velocity_df = velocity_df.fillna(0)  # Fill with zeros
    
    coeff = torch.tensor(coefficients[str(order)], device='cuda', dtype=torch.float32)
    h = torch.tensor(h, device='cuda', dtype=torch.float32)
    
    # Convert locations to a PyTorch tensor and move it to the GPU
    locations_tensor = torch.tensor(locations[locations_index].values, device='cuda', dtype=torch.float32)
    
    def apply_central_diff(row_index):
        if row_index < order or row_index >= len(locations) - order:
            return torch.zeros(len(vals_index), device='cuda')
        else:
            velocities = []
            for i, loc in enumerate(locations_index):
                segment = locations_tensor[row_index-order:row_index+order+1, i]
                velocity = torch.dot(coeff, segment) / h
                velocities.append(velocity)
            return torch.tensor(velocities, device='cuda')
    
    # Apply the central difference calculation for each valid row
    results = [apply_central_diff(i) for i in range(len(locations))]
    results = torch.stack(results)
    
    # Convert results back to a DataFrame and move it to CPU
    results_df = pd.DataFrame(results.cpu().numpy(), columns=vals_index, index=locations.index)
    
    # Concatenate the original locations DataFrame with the velocity DataFrame
    locations_with_velocity = pd.concat([locations_with_velocity, results_df], axis=1)
    
    return locations_with_velocity


def ewm(data, alpha):
    """
    Calculate the Exponential Weighted Moving Average (EWMA) of a data series.

    Parameters:
    data (array-like): The input data series.
    alpha (float): The smoothing factor (0 < alpha <= 1).

    Returns:
    np.ndarray: The EWMA of the data series.
    """
    data = np.asarray(data)
    ewma = np.zeros_like(data)
    ewma[0] = data[0]  # The first value is the same as the data series
    
    for t in range(1, len(data)):
        ewma[t] = alpha * data[t] + (1 - alpha) * ewma[t-1]
    
    return ewma

def hybrid_model_training(config, train_loader, encoder_train_loader, model, reconstructor, device, num_epochs_initial=100, num_epochs_fine_tune=50):
    
    # Initialize the Reconstructing Transformer model
    reconstructor = reconstructor.to(device)
    model = model.to(device)

    optimizer = torch.optim.Adam(reconstructor.parameters(), lr=1e-3)
    criterion = RMSELoss()

    # Step 1: Initial training on the distribution of time series A
    for epoch in range(num_epochs_initial):
        reconstructor.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon_batch = reconstructor(batch)
            loss = criterion(recon_batch, batch)
            loss.backward()
            optimizer.step()
        print(f"Initial Training Epoch [{epoch+1}/{num_epochs_initial}], Loss: {loss.item()}")

    # Step 2: Fine-tuning on the encoder transformer's outputs
    for epoch in range(num_epochs_fine_tune):
        reconstructor.train()
        for encoder_outputs, targets in encoder_train_loader:
            encoder_outputs = encoder_outputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            adjusted_outputs = reconstructor(encoder_outputs)
            loss = criterion(adjusted_outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"Fine-tuning Epoch [{epoch+1}/{num_epochs_fine_tune}], Loss: {loss.item()}")

    return reconstructor