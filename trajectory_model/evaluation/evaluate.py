import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.metrics import RMSELoss, TSS, RSS, R2
from scipy.signal import savgol_filter
import time
from torch.nn import MSELoss
import torch

def test_model(model, config ,
                # criterion ,
                data_loader, 
                scalar ,
                device='cpu',
                make_pdf=True,
                epoch = 0,
                task = 'train',
                critic = None,
                vae = None):


        criterion = MSELoss()

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

                outputs = model(inputs)
                outputs =outputs[config["iTransformer_pred_length"]]
                loss = criterion(outputs, targets)
                outputs = outputs.cpu().detach().numpy()
                targets = targets.cpu().detach().numpy()
                TSS_losses.append(TSS(targets))
                RSS_losses.append(RSS(targets,outputs))


                batchsize,pred_len,variabel_size = inputs.shape
                unnorm_outputs = scalar.inverse_transform(outputs.reshape((-1,variabel_size)))
                unnorm_targets = scalar.inverse_transform(targets.reshape((-1,variabel_size)))
                
                location_error = (np.abs(unnorm_outputs - unnorm_targets).sum(axis=0))/inputs.size(0)
                
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
                sum_location_error += location_error

            # # Apply Savitzky-Golay Filter
            window_size = 100
            poly_order = 5
            # predsToPlot = savgol_filter(predsToPlot,deriv=0, window_length=window_size, polyorder=poly_order,axis=0)
            # firstOrder_derivative_predsToPlot = savgol_filter(predsToPlot,deriv=1,delta=30.0, window_length=window_size, polyorder=poly_order,axis=0)
            # predsToPlot += firstOrder_derivative_predsToPlot*200 

            # Reshape the arrays to separate the coordinates of each point

            true_reshaped = targetsToPlot.reshape(-1, 3, 3)  # shape will be [n, 3, 3]
            pred_reshaped = predsToPlot.reshape(-1, 3, 3)  # shape will be [n, 3, 3]
            # Calculate the Euclidean distance for each corresponding point
            distances = np.linalg.norm(true_reshaped - pred_reshaped, axis=2)

            # Calculate the mean error
            mean_error = np.mean(distances,axis=0)

            avg_loss = total_loss/len(data_loader)
            avg_iter_time = total_time/len(data_loader)
            
            avg_location_error = sum_location_error/len(data_loader)


            title = (
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
                legends = config["label_index"] # ['MEx','MEY' ...]
                for i in range(6):
                    plt.subplot(6,1,(i+1))
                    idx = i
                    plt.plot(targetsToPlot[:,idx+3:idx+3+1], linestyle='-')
                    plt.legend([legends[i+3]])
                    plt.plot(predsToPlot[:,idx+3:idx+3+1], linestyle='--')
                    # Set the x-axis limits
                    # plt.ylim(-0.3, 0.7)

                # plt.suptitle(title)


                dir_path = './results/' + "PosePredictor"

                # Check if the directory exists
                if not os.path.exists(dir_path):
                    # Create the directory if it does not exist
                    os.makedirs(dir_path)
                if task == "test":
                    name =  "PosePredictor" +'test' + '.pdf'
                    name_txt = "PosePredictor" +'test' + '.txt'
                else:
                    name =  "PosePredictor" +'_epoch_'+ str(epoch)  + '.pdf'
                    name_txt = "PosePredictor" +'_epoch_'+ str(epoch)  + '.txt'
                with open(os.path.join(dir_path,name_txt),'w') as f :
                    f.write(title)
                    f.write(str(config))
                plt.savefig(os.path.join(dir_path,name))
                plt.close()
            
        avg_euc_end_effector_error = mean_error[-1]
        max_euc_end_effector_error = distances.max()
        R2_score = R2(TSS_losses,RSS_losses)
        return avg_loss,avg_critic_loss,avg_iter_time, avg_location_error,avg_euc_end_effector_error,max_euc_end_effector_error,R2_score

