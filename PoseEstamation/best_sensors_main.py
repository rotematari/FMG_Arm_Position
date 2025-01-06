from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
from data_processes.data_loader import FMGPoseDataset_important_sensors
from utils.utils import load_yaml_config ,set_device, set_seed
from train.train import train
from evaluation.evaluate import test_model
from models.get_model import get_model
import numpy as np

def main(score_config=None,score=None):
    #                     "dropout": 0.12555232252228307,
    #                     "learning_rate": 0.005236027155259325,
    #                     "weight_decay": 0.007059887693062261,
    #                     "warmup_length": 0.3363629602379294
    # Load the config file
    config = load_yaml_config('config.yaml')
    # config["dropout"] = score_config["dropout"]
    # config["learning_rate"] = score_config["learning_rate"]
    # config["weight_decay"] = score_config["weight_decay"]
    # config["warmup_length"] = score_config["warmup_length"]
    config["dropout"] = 0.125
    config["learning_rate"] = 0.0052
    config["weight_decay"] = 0.007
    config["warmup_length"] = 0.336
    config["experiment_name"] = f"best_sensors_{score}_"
    
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
    # score_config = {0.5:{   
    #                     "dropout": 0.12555232252228307,
    #                     "learning_rate": 0.005236027155259325,
    #                     "weight_decay": 0.007059887693062261,
    #                     "warmup_length": 0.3363629602379294,},
    #             1.0:{   
    #                     "dropout": 0.1359152103463657,
    #                     "learning_rate": 0.0022988673236602293,
    #                     "weight_decay": 0.00128666713660346,
    #                     "warmup_length": 0.3337615171403628,},
    #             1.5:{        
    #                 "dropout": 0.1284130532073927,
    #                 "learning_rate": 0.007286961220815371,
    #                 "weight_decay": 0.008981391573530513,
    #                 "warmup_length": 0.38870864242651176,},
    #             2.0:{       
    #                 "dropout": 0.12577016542294217,
    #                 "learning_rate": 0.007323101165546835,
    #                 "weight_decay": 0.0064118189664166105,
    #                 "warmup_length": 0.3887212742576327,},
    #             2.5:{        
    #                 "dropout": 0.10024130445864891,
    #                 "learning_rate": 0.008982436003737935,
    #                 "weight_decay": 0.009014138765916972,
    #                 "warmup_length": 0.3633101457273268,}

    #     }
    # for score in [0.5,1.0,1.5,2.0,2.5]:
    #     main(score_config[score],score=score)
    main(score=0.5)