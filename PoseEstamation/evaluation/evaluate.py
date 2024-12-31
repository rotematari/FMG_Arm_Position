import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.metrics import RMSELoss, TSS, RSS, R2
from utils.utils import save_to_csv
from scipy.signal import savgol_filter
import time
from torch.nn import MSELoss
import torch

def test_model(model, config ,
                # criterion ,
                data_loader, 
                label_scaler ,
                device='cpu',
                make_pdf = True,
                epoch = 0,
                task = 'train',
                criterion_elbow = RMSELoss(),
                criterion_wrist = RMSELoss(),
                loss_weight = 0.1,
                critic = None,
                vae = None,
                number_of_data_samples=None,
                experiment=None):

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
            for i,(inputs, targets) in train_iterator:

                inputs = inputs.to(device=device)
                targets = targets.to(device=device)

                
                start_time = time.time()

                # outputs = model(inputs)
                elbow_outputs,wrist_outputs,_ = model(inputs)
                outputs = torch.cat((elbow_outputs, wrist_outputs), dim=1)
                # outputs =outputs[config["iTransformer_pred_length"]]
                # loss = criterion(outputs, targets)
                elbow_targets = targets[:,:3]
                wrist_targets = targets[:,3:]
                # elbow_outputs = outputs[:,:3]
                # wrist_outputs = outputs[:,3:]
                loss = loss_weight*criterion_elbow(elbow_outputs,elbow_targets)+ (1-loss_weight)*criterion_wrist(wrist_outputs,wrist_targets)

                outputs = outputs.cpu().detach().numpy()
                targets = targets.cpu().detach().numpy()
                TSS_losses.append(TSS(targets))
                RSS_losses.append(RSS(targets,outputs))

                unnorm_outputs = label_scaler.inverse_transform(outputs)
                unnorm_targets = label_scaler.inverse_transform(targets)

                
                location_error = (np.abs(unnorm_outputs - unnorm_targets).sum(axis=0))/inputs.size(0)
                sum_location_error += location_error

                if i == 0:
                    predsToPlot = unnorm_outputs
                    targetsToPlot = unnorm_targets
                    inputsToSave = inputs.cpu().detach().numpy()

                predsToPlot = np.concatenate((predsToPlot,unnorm_outputs))
                targetsToPlot = np.concatenate((targetsToPlot,unnorm_targets))
                inputsToSave = np.concatenate((inputsToSave,inputs.cpu().detach().numpy()))

                end_time = time.time()
                total_time += (end_time - start_time)
                total_loss += loss.item()
                        # # Apply Savitzky-Golay Filter
            # window_size = 50
            # poly_order = 5
            # predsToPlot = savgol_filter(predsToPlot,deriv=0, window_length=window_size, polyorder=poly_order,axis=0)
            # firstOrder_derivative_predsToPlot = savgol_filter(predsToPlot,deriv=1,delta=30.0, window_length=window_size, polyorder=poly_order,axis=0)
            # predsToPlot += firstOrder_derivative_predsToPlot*300 
            # save all preds and targets 
            save_to_csv(targetsToPlot)
            save_to_csv(predsToPlot,model_name=model.name)
            # Reshape the arrays to separate the coordinates of each point
            true_reshaped = targetsToPlot.reshape(-1, 2, 3)  # shape will be [n, 2, 3]
            pred_reshaped = predsToPlot.reshape(-1, 2, 3)  # shape will be [n, 2, 3]
            # Calculate the Euclidean distance for each corresponding point

            distances = np.linalg.norm(true_reshaped - pred_reshaped, axis=2)

            # Calculate the mean error
            mean_error = np.mean(distances,axis=0)
            std = np.std(distances,axis=0)

            avg_loss = total_loss/len(data_loader)
            avg_iter_time = total_time/len(data_loader)
            avg_location_error = sum_location_error/len(data_loader)
            # Calculate R² score
            R2_score = R2(TSS_losses, RSS_losses)

            # Update the title with the new metrics
            title = (
                f"RMSE Avg Loss: {round(avg_loss, 5)}\n"
                f"Avg Iter Time: {round(avg_iter_time, 5)}\n"
                f"Avg Euclidean Wrist Error: {round(mean_error[-1], 5)}\n"
                f"Euclidean Wrist STD: {round(std[-1], 5)}\n"
                f"Avg Euclidean Elbow Error: {round(mean_error[0], 5)}\n"
                f"Euclidean Elbow STD: {round(std[0], 5)}\n"
                f"Max Euclidean End Effector Error: {round(distances.max(), 5)}\n"
                f"Avg Location Error: {avg_location_error}\n"
                f"R² Score: {round(R2_score, 5)}\n"
            )
            
            # to save df 
            # np.save('inputs.npy',inputsToSave)

            # df_inputs = pd.DataFrame(inputsToSave,columns=config.fmg_index)
            # df_true = pd.DataFrame(targetsToPlot,columns=config.label_index)
            # df_preds = pd.DataFrame(predsToPlot,columns=config.label_index)
            # df_true.to_csv('df_true.csv')
            # df_preds.to_csv('df_preds.csv')
            # df_inputs.to_csv('df_inputs.csv')
            
            plot_and_save_results(config, targetsToPlot, predsToPlot, task,title, epoch=epoch,number_of_data_samples=number_of_data_samples,experiment=experiment)
            

        avg_elbow_error = mean_error[0]
        elbow_std = std[0]
        avg_euc_end_effector_error = mean_error[-1]
        wrist_std = std[-1]
        max_euc_end_effector_error = distances.max()
        R2_score = R2(TSS_losses,RSS_losses)
        return avg_loss,avg_critic_loss,avg_iter_time, avg_location_error,avg_euc_end_effector_error,max_euc_end_effector_error,R2_score,avg_elbow_error,wrist_std,elbow_std

def plot_and_save_results(config, targetsToPlot, predsToPlot, task, title="",epoch=None, number_of_data_samples=None,experiment=None):
    """
    Plots the target and predicted values for comparison and saves the plot as a PDF.

    Parameters:
    - config: Dictionary containing configuration details.
    - targetsToPlot: Array of target values to plot.
    - predsToPlot: Array of predicted values to plot.
    - task: Task name ('train' or 'test') for saving the file.
    - epoch: Current epoch number (optional).
    - title: Title for the plot and accompanying text file.
    """
    
    legends = [
       # 'MCx','MCy', 'MCz',
       # 'MSx', 'MSy', 'MSz',
       'MEx', 'MEy', 'MEz',
       'MWx', 'MWy', 'MWz'
]       # e.g., ['MEx', 'MEY', ...]
    
    for i in range(6):
        plt.subplot(6, 1, i + 1)
        plt.plot(targetsToPlot[:, i:i+1], linestyle='-')
        plt.legend([legends[i]])
        plt.plot(predsToPlot[:, i:i+1], linestyle='--')
    if number_of_data_samples is not None:
        dir_path = './' + config["data_test_path"] + '/' + str(number_of_data_samples) + '_data_sample/ex_' + str(experiment)
    else:
        # Directory path for saving results
        dir_path = './results/' + config["model"]

    # Create directory if it doesn't exist
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Determine file name based on task and epoch
    if task == "test":
        name = config["model"] + 'test' + '.pdf'
        name_txt = config["model"] + 'test' + '.txt'

    else:
        name = config["model"] + '_epoch_' + str(epoch) + '.pdf'
        name_txt = config["model"] + '_epoch_' + str(epoch) + '.txt'

    # Write title and config to a text file
    with open(os.path.join(dir_path, name_txt), 'w') as f:
        f.write(title + "\n")
        f.write(str(config))

    # Save the plot
    plt.savefig(os.path.join(dir_path, name))
    plt.close()

