import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.nn import MSELoss
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from utils.metrics import RMSELoss, TSS, RSS, R2
from evaluation.evaluate import test_model
import time
from copy import deepcopy
from os.path import join
import os 
import csv
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))

def train_unsupervised(config,model,optimizer, train_loader,device='cpu'):
    num_epochs = config["epochs"]
    # num_epochs = 30
    
    # pred_length = config["iTransformer_pred_length"]
    
    
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = int(0.2 * num_training_steps)  # 10% warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps,
        num_cycles = 0.5
    )
    criterion = RMSELoss()
    for epoch in range(num_epochs):
        # Initialize the epoch loss and accuracy
        train_loss = 0
        model.train()

        # Wrap your training loop iterator with tqdm
        train_iterator = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training un supervised")
        # Train on the training set
        for i,(inputs,targets) in train_iterator:
            
            inputs = inputs.to(device=device)
            targets = targets.to(device=device)

            optimizer.zero_grad()

            elbow_outputs,wrist_outputs,fmg = model(inputs)

            loss = criterion(fmg,inputs)
            loss.backward()
            train_loss +=  loss.item()
            optimizer.step()

            if scheduler: scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            train_iterator.set_description(f"Epoch [{epoch}/{num_epochs}] Train Loss: {(train_loss/(i+1)):.4f} LR: {current_lr:.6f}")

def train(config,model,optimizer, train_loader,val_loader ,label_scaler,feature_scaler, device='cpu',wandb_run=None):
    
    num_epochs = config["epochs"]
    # pred_length = config["iTransformer_pred_length"]
    
    
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = int(0.2 * num_training_steps)  # 10% warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps,
        num_cycles = 0.5
    )

    criterion_elbow = RMSELoss()
    criterion_wrist = RMSELoss()
    train_losses = []
    val_losses = []
    TSS_losses = []
    RSS_losses = []
    best_val_loss = 10
    best_wrist_error = 30
    print("training starts")
    train_unsupervised(config=config,model=model,optimizer=optimizer,train_loader=train_loader,device=device)
    for epoch in range(num_epochs):
        # Initialize the epoch loss and accuracy
        train_loss = 0
        model.train()
        # modelV.train()
        train_loss, train_loss_loc, train_loss_v = 0, 0, 0

        # Wrap your training loop iterator with tqdm
        train_iterator = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")
        # Train on the training set
        for i,(inputs,targets) in train_iterator:


            inputs = inputs.to(device=device)
            targets = targets.to(device=device)

            optimizer.zero_grad()

            elbow_outputs,wrist_outputs,_ = model(inputs)
            outputs = torch.cat((elbow_outputs, wrist_outputs), dim=1)
            # outputs =outputs[pred_length]
            # loss = criterion(outputs, targets)
            elbow_targets = targets[:,:3]
            wrist_targets = targets[:,3:]
            # elbow_outputs = outputs[:,:3]
            # wrist_outputs = outputs[:,3:]
            loss_weight = 0.1
            loss = loss_weight*criterion_elbow(elbow_outputs,elbow_targets)+ (1-loss_weight)*criterion_wrist(wrist_outputs,wrist_targets)
            loss.backward()



            train_loss +=  loss.item()
            TSS_losses.append(TSS(targets.cpu().detach().numpy()))
            RSS_losses.append(RSS(targets.cpu().detach().numpy(),outputs.cpu().detach().numpy()))

            optimizer.step()
            if scheduler: scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            train_iterator.set_description(f"Epoch [{epoch}/{num_epochs}] Train Loss: {(train_loss/(i+1)):.4f} R^2:{R2(TSS_losses,RSS_losses):.3f} LR: {current_lr:.6f}")

        train_loss /= len(train_loader)
        train_losses.append(train_loss)


        val_loss,avg_critic_loss,avg_iter_time, avg_location_error,avg_euc_end_effector_error,max_euc_end_effector_error,R2_score,avg_elbow_error,wrist_std,elbow_std = test_model(
            model=model,
            config=config,
            data_loader=val_loader,
            label_scaler=label_scaler,
            device=device,
            epoch=epoch,
            task='validate',
            make_pdf= True,
            loss_weight=loss_weight
        )


        print(f'Epoch: {epoch} Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f} R2 {R2_score:.4f} \n Avarege Euclidian End Effector Error: {avg_euc_end_effector_error} Max Euclidian End Effector Error:{max_euc_end_effector_error} time for one iteration {1000*avg_iter_time:.4f} ms \n ----------')
        # Save the validation loss 
        val_losses.append(val_loss)
        
        # if config.wandb_on:
        #     # log metrics to wandb
        #     wandb.log({"Train_Loss": train_loss, "Val_loss": val_loss, "Val_Max_Euclidian_End_Effector_Error" : max_euc_end_effector_error , "Val_Avarege_Euclidian_End_Effector_Error": avg_euc_end_effector_error})

        if( avg_euc_end_effector_error < best_wrist_error):
        # if( best_val_loss > val_loss):
            best_wrist_error = avg_euc_end_effector_error
            # best_val_loss = val_loss
            time_stamp = time.strftime("%d_%m_%H_%M", time.gmtime())
            filename = model.name + '_epoch_' +str(epoch)+'_date_'+time_stamp + '.pt'
            best_model_checkpoint_path = join(config["model_path"],filename)
            best_model_checkpoint = {
                'epoch': epoch,
                'model_state_dict': deepcopy(model.state_dict()),
                'optimizer_state_dict': deepcopy(optimizer.state_dict()),
                # 'scaler_state_dict': scaler.state_dict(),
                'loss': val_loss,
                'config': config if not config["wandb_on"] else 'in_wandb' ,
                }
            # Extract the state of the label_scaler
            label_scaler_state = {
                'mean': label_scaler.mean_.tolist(),
                'var': label_scaler.var_.tolist(),
                'scale': label_scaler.scale_.tolist(),
                'n_samples_seen': label_scaler.n_samples_seen_
            }
            # Extract the state of the feature_scaler
            feature_scaler_state = {
                'mean': feature_scaler.mean_.tolist(),
                'var': feature_scaler.var_.tolist(),
                'scale': feature_scaler.scale_.tolist(),
                'n_samples_seen': feature_scaler.n_samples_seen_
            }
                        # Extract the state of the label_scaler

            # Add the scaler state to your checkpoint dictionary
            best_model_checkpoint['label_scaler_state'] = label_scaler_state
            best_model_checkpoint['feature_scaler_state'] = feature_scaler_state

    torch.save(best_model_checkpoint,best_model_checkpoint_path)

    print(f"model {filename} saved ")
    
    return 


def save_experiment_results(file_path, experiment_data):
    """
    Save the results of the experiment to a CSV file.
    
    Parameters:
    - file_path: The path to the CSV file.
    - experiment_data: A dictionary containing the experiment data.
    """
    # Define CSV headers (make sure the order of fields is the same as the keys in experiment_data)
    headers = [
        'number_of_data_samples', 'experiment_number', 'rmse_loss', 
        'wrist_error','wrist_std', 
        'avg_elbow_error','elbow_std', 'selected_files'
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

def train_for_data_test(config,model,optimizer, train_loader,val_loader ,label_scaler,feature_scaler, 
                        device='cpu',wandb_run=None,number_of_data_samples=1,experiment=1,):
    
    num_epochs = config["epochs"]
    # pred_length = config["iTransformer_pred_length"]
    
    
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = int(0.2 * num_training_steps)  # 10% warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps,
        num_cycles = 0.5
    )

    criterion_elbow = RMSELoss()
    criterion_wrist = RMSELoss()
    train_losses = []
    val_losses = []
    TSS_losses = []
    RSS_losses = []
    best_val_loss = 10
    best_wrist_error = 30
    print("training starts")
    train_unsupervised(config=config,model=model,optimizer=optimizer,train_loader=train_loader,device=device)
    for epoch in range(num_epochs):
        # Initialize the epoch loss and accuracy
        train_loss = 0
        model.train()
        # modelV.train()
        train_loss, train_loss_loc, train_loss_v = 0, 0, 0

        # Wrap your training loop iterator with tqdm
        train_iterator = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")
        # Train on the training set
        for i,(inputs,targets) in train_iterator:


            inputs = inputs.to(device=device)
            targets = targets.to(device=device)

            optimizer.zero_grad()

            elbow_outputs,wrist_outputs,_ = model(inputs)
            outputs = torch.cat((elbow_outputs, wrist_outputs), dim=1)
            # outputs =outputs[pred_length]
            # loss = criterion(outputs, targets)
            elbow_targets = targets[:,:3]
            wrist_targets = targets[:,3:]
            # elbow_outputs = outputs[:,:3]
            # wrist_outputs = outputs[:,3:]
            loss_weight = 0.1
            loss = loss_weight*criterion_elbow(elbow_outputs,elbow_targets)+ (1-loss_weight)*criterion_wrist(wrist_outputs,wrist_targets)
            loss.backward()



            train_loss +=  loss.item()
            TSS_losses.append(TSS(targets.cpu().detach().numpy()))
            RSS_losses.append(RSS(targets.cpu().detach().numpy(),outputs.cpu().detach().numpy()))

            optimizer.step()
            if scheduler: scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            train_iterator.set_description(f"Epoch [{epoch}/{num_epochs}] Train Loss: {(train_loss/(i+1)):.4f} R^2:{R2(TSS_losses,RSS_losses):.3f} LR: {current_lr:.6f}")

        train_loss /= len(train_loader)
        train_losses.append(train_loss)


        val_loss,val_critic_loss,avg_iter_time, avg_location_error,avg_euc_end_effector_error,max_euc_end_effector_error,R2_score,avg_elbow_error,wrist_std,elbow_std = test_model(
            model=model,
            config=config,
            data_loader=val_loader,
            label_scaler=label_scaler,
            device=device,
            epoch=epoch,
            task='validate',
            make_pdf= True,
            loss_weight=loss_weight,
            number_of_data_samples = number_of_data_samples,
            experiment=experiment
        )


        print(f'Epoch: {epoch} Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f} R2 {R2_score:.4f} \n Avg Wrist Error: {avg_euc_end_effector_error:.4f} Wrist std: {wrist_std} Max Euclidian End Effector Error:{max_euc_end_effector_error:.4f} Avg Elbow Error: {avg_elbow_error:.4f} Elbow std: {elbow_std:.4f} time for one iteration {1000*avg_iter_time:.4f} ms \n ----------')
        # Save the validation loss 
        val_losses.append(val_loss)
        
        # if config.wandb_on:
        #     # log metrics to wandb
        #     wandb.log({"Train_Loss": train_loss, "Val_loss": val_loss, "Val_Max_Euclidian_End_Effector_Error" : max_euc_end_effector_error , "Val_Avarege_Euclidian_End_Effector_Error": avg_euc_end_effector_error})

        if( avg_euc_end_effector_error < best_wrist_error):
        # if( best_val_loss > val_loss):
            best_wrist_error = avg_euc_end_effector_error
            # best_val_loss = val_loss
            time_stamp = time.strftime("%d_%m_%H_%M", time.gmtime())
            filename = model.name + str(number_of_data_samples) + '_epoch_' +str(epoch)+'_date_'+time_stamp + '.pt'
            dir_name = join(config["data_test_path"],str(number_of_data_samples) + '_data_sample/ex_' + str(experiment))
            best_model_checkpoint_path = join(dir_name,filename)
            # Create directory if it doesn't exist
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            best_model_checkpoint = {
                'epoch': epoch,
                'model_state_dict': deepcopy(model.state_dict()),
                'optimizer_state_dict': deepcopy(optimizer.state_dict()),
                # 'scaler_state_dict': scaler.state_dict(),
                'loss': val_loss,
                'config': config if not config["wandb_on"] else 'in_wandb' ,
                }
            # Extract the state of the label_scaler
            label_scaler_state = {
                'mean': label_scaler.mean_.tolist(),
                'var': label_scaler.var_.tolist(),
                'scale': label_scaler.scale_.tolist(),
                'n_samples_seen': label_scaler.n_samples_seen_
            }
            # Extract the state of the feature_scaler
            feature_scaler_state = {
                'mean': feature_scaler.mean_.tolist(),
                'var': feature_scaler.var_.tolist(),
                'scale': feature_scaler.scale_.tolist(),
                'n_samples_seen': feature_scaler.n_samples_seen_
            }
                        # Extract the state of the label_scaler

            # Add the scaler state to your checkpoint dictionary
            best_model_checkpoint['label_scaler_state'] = label_scaler_state
            best_model_checkpoint['feature_scaler_state'] = feature_scaler_state

    torch.save(best_model_checkpoint,best_model_checkpoint_path)

    print(f"model {filename} saved ")

            # Prepare experiment data to save to CSV
    experiment_data = {
        'number_of_data_samples': number_of_data_samples,
        'experiment_number': experiment,
        'rmse_loss': round(val_loss, 5),
        'wrist_error': round(avg_euc_end_effector_error, 5),
        'wrist_std': round(wrist_std, 5),
        'avg_elbow_error': round(avg_elbow_error, 5),
        'elbow_std': round(elbow_std, 5),
        'selected_files': ', '.join([file for file in train_loader.dataset.selected_files])  # Assuming the selected files are stored in the dataset
    }

        # Save results to CSV
    save_experiment_results(join(config["data_test_path"],"csv_data_experiment_results.csv"), experiment_data)
    return 
