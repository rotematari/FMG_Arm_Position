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

def train(config,model,optimizer, train_loader,val_loader ,scalar, device='cpu',wandb_run=None):
    
    num_epochs = config["num_epochs"]
    pred_length = config["iTransformer_pred_length"]
    
    
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = int(0.2 * num_training_steps)  # 10% warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps,
        num_cycles = 0.5
    )

    criterion = MSELoss()
    train_losses = []
    val_losses = []
    TSS_losses = []
    RSS_losses = []
    best_val_loss = 10
    print("training starts")

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

            outputs = model(inputs)
            outputs =outputs[pred_length]
            loss = criterion(outputs, targets)
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


        val_loss,val_critic_loss,avg_iter_time, avg_location_error,avg_euc_end_effector_error,max_euc_end_effector_error,R2_score = test_model(
            model=model,
            config=config,
            data_loader=val_loader,
            scalar=scalar,
            device=device,
            epoch=epoch,
            task='validate',
            make_pdf= wandb_run,
        )


        print(f'Epoch: {epoch} Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f} R2 {R2_score:.4f} \n Avarege Euclidian End Effector Error: {avg_euc_end_effector_error} Max Euclidian End Effector Error:{max_euc_end_effector_error} time for one iteration {1000*avg_iter_time:.4f} ms \n ----------')
        # Save the validation loss 
        val_losses.append(val_loss)
        
        # if config.wandb_on:
        #     # log metrics to wandb
        #     wandb.log({"Train_Loss": train_loss, "Val_loss": val_loss, "Val_Max_Euclidian_End_Effector_Error" : max_euc_end_effector_error , "Val_Avarege_Euclidian_End_Effector_Error": avg_euc_end_effector_error})

        # if( avg_euc_end_effector_error < best_wrist_error):
        if( best_val_loss > val_loss):
            # best_wrist_error = avg_euc_end_effector_error
            best_val_loss = val_loss
            time_stamp = time.strftime("%d_%m_%H_%M", time.gmtime())
            filename = "PosePredictor" + '_epoch_' +str(epoch)+'_date_'+time_stamp + '.pt'
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
            scaler_state = {
                'mean': scalar.mean_.tolist(),
                'var': scalar.var_.tolist(),
                'scale': scalar.scale_.tolist(),
                'n_samples_seen': scalar.n_samples_seen_
            }
                        # Extract the state of the label_scaler

            # Add the scaler state to your checkpoint dictionary
            best_model_checkpoint['scaler_state'] = scaler_state


    torch.save(best_model_checkpoint,best_model_checkpoint_path)

    print(f"model {filename} saved ")
    
    return 




