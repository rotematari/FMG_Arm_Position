from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
from data_processes.data_loader import FMGPoseDataset_important_sensors
from utils.utils import load_yaml_config ,set_device, set_seed
from train.train import train
from evaluation.evaluate import test_model
from models.get_model import get_model
import numpy as np

def main(score=3.0):
        # Load the config file
    config = load_yaml_config('config.yaml')

    feature_impotance_scores = [1.39, 4.93, 12.82, 0.99, 3.37, 7.88, 0.5, 5.88,
                                9.52, 4.94, 0.89, 1.25, 0.07, 0.01, 0.66, 1.58, 
                                0.99, 2.36, 1.63, 6.49, 8.23, 1.07, 0.94, 4.65,
                                0.62, 1.18, 1.07, 1.16, 1.11, 0.91, 0.49, 13.35]
    
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

    config["input_size"] = len(sensors_to_use)
    print("-----------------")
    print(f"Using {len(sensors_to_use)} sensors")
    print("-----------------")
    # Set the seed
    set_seed(config["seed"])
    device = set_device()

    # Load the dataset
    trainDataset = FMGPoseDataset_important_sensors(config,mode='train',sensor_to_use=sensors_to_use)
    testDataset = FMGPoseDataset_important_sensors(config,mode='test',sensor_to_use=sensors_to_use,feature_scalar=trainDataset.feature_scalar,label_scalar= trainDataset.label_scalar)
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


if __name__ == '__main__':
    main()