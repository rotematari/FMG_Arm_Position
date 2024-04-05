import torch
import argparse
import yaml
import wandb
import os 
import numpy as np
import random

# Change the current working directory to the directory of the script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from data.data_processing import DataProcessor
from models.models import CNNLSTMModel , TransformerModel,DLinear , DecompTransformerModel,Conv2DLSTMAttentionModel,TimeSeriesTransformer
# from models.Autoformer.models.Autoformer import Model as Autoformer
from models.PatchTST import Model as PatchTST
from utils.utils import train, test_model, plot_results,print_model_size, plot_losses
from pytorch_tcn import TCN

def set_device(config):
    if torch.cuda.is_available():

        device = torch.device(f"cuda:{config.cuda}")
        print(f"cuda:{config.cuda}\n")
        print(f"Running on the CUDA device: {torch.cuda.get_device_name(config.cuda)}")
    else:
        device = torch.device("cpu")
        print("Running on CPU")
    return device


def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file.')
    parser.add_argument('--cuda', type=int, help='0-3 choose cuda to use in zeus')
    parser.add_argument('--n_layer', type=int, help='The number of hidden layers.')
    parser.add_argument('--lstm_hidden_size', type=int, help='The size of each hidden layer.')
    parser.add_argument('--lstm_num_layers', type=int, help='The number of layers.')
    parser.add_argument('--dropout', type=float, help='The dropout rate for each hidden layer.')
    parser.add_argument('--learning_rate', type=float, help='The learning rate for the optimizer.')
    parser.add_argument('--num_epochs', type=int, help='The number of epochs to train for.')
    parser.add_argument('--weight_decay', type=float, help='The weight decay for the optimizer.')
    parser.add_argument('--batch_size', type=int, help='The size of batches.')
    parser.add_argument('--window_size', type=int, help='The size of batches.')
    parser.add_argument('--sequence_length', type=int, help='The sequence length.')
    parser.add_argument('--model', type=str, help='network to use')
    parser.add_argument('--norm', type=str, help='normalization technic')
    parser.add_argument('--n_head', type=str, help='n_attentiopn head')
    parser.add_argument('--cnn_hidden_size', type=int, help='Hidden size for the CNN layers')
    parser.add_argument('--cnn_kernel_size', type=int, help='Kernel size for the CNN layers')
    parser.add_argument('--drop_epoch', type=int, help='Epoch at which to drop the learning rate')
    parser.add_argument('--loss_func', type=str, choices=['MSELoss', 'RMSELoss'], help='Loss function to use')
    parser.add_argument('--maxpoll_kernel_size', type=int, help='Kernel size for the max pooling layers')
    parser.add_argument('--norm_labels', type=bool, help='Whether to normalize labels')
    parser.add_argument('--stable_lr', type=float, help='Stable learning rate to use after drop_epoch')
    parser.add_argument('--kernelsize_tcn', type=int, help='Kernel size for TCN')
    parser.add_argument('--num_channels', help='List of number of channels (e.g., [28, 28])')
    parser.add_argument('--cnn2d_kernel_size', type=int, help='Kernel size for 2D CNN.')
    parser.add_argument('--cnn2dlstm_dropout', type=float, help='Dropout rate for CNN2DLSTM.')
    parser.add_argument('--conv2d_hidden_sizes', nargs='+', type=int, help='List of hidden sizes for Conv2D.')
    parser.add_argument('--conv2d_n_heads', type=int, help='Number of heads for Conv2D.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    return argparse.Namespace(**config)


def main():
    
    torch.manual_seed(42)  # Replace 0 with your desired seed
    # If using CUDA:
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)  # For multi-GPU setups
    np.random.seed(42)
    random.seed(0)
    config = parse_args()

    device = set_device(config)


    run = None

    if config.wandb_on:
        wandb_run = wandb.init(project="tune_fmg_conv2dlstm", config=config)
        # wandb.config.update(allow_val_change=True)
        
        config = wandb.config
        # config.update(allow_val_change=True)
        wandb.define_metric("Val_loss", summary="min")
        wandb.define_metric("Train_loss", summary="min")
        wandb.define_metric("Val_Avarege_Euclidian_End_Effector_Error", summary="min")
        wandb.define_metric("Val_Max_Euclidian_End_Effector_Error", summary="min")

    data_processor = DataProcessor(config)
    data_processor.load_data()
    data_processor.preprocess_data()

    if config.wandb_on:
        config.update({'num_labels':len(data_processor.label_index)},allow_val_change=True)
    else:
        config.num_labels = len(data_processor.label_index)

    if config.plot_data:
        data_processor.plot(from_indx=100000, to_indx=105000)
    
    if config.pre_trained:
        check_point = torch.load(config.best_model)
        # config = check_point['config']

    train_loader, test_loader = data_processor.get_data_loaders()

    # for model in config.models_to_train:
    #     config.model = model
    if config.model == 'CNN_LSTMModel':
        model = CNNLSTMModel(config).to(device)
    if config.model == 'Conv2DLSTMAttentionModel':
        model = Conv2DLSTMAttentionModel(config).to(device)
    elif config.model == 'TransformerModel':
        model = TransformerModel(config).to(device)
    elif config.model == 'TimeSeriesTransformer':
        model = TimeSeriesTransformer(config).to(device)
    elif config.model == 'PatchTST':
        model = PatchTST(config).to(device)
    elif config.model == 'DLinear':
        model = DLinear(config).to(device)
    elif config.model == 'TCN':
        model = TCN( num_inputs = config.input_size,
                        num_channels = config.num_channels,
                        kernel_size =  config.kernelsize_tcn,
                        dilations = [2 ** i for i in range(len(config.num_channels))],
                        dilation_reset = 8,
                        dropout = config.dropout,
                        causal =  True,
                        use_norm = 'layer_norm',
                        activation = 'relu',
                        kernel_initializer = 'xavier_uniform',
                        use_skip_connections = True,
                        input_shape = 'NLC').to(device)
        
    elif config.model == 'DecompTransformerModel':
        model = DecompTransformerModel(config).to(device)
    
    # if config.wandb_on:
        # config.sequence_length = 2 ** config.sequence_length
        # config.update(config)
    print(model)
    print_model_size(model)
    print(f"Train_len {len(train_loader)*config.batch_size} samples, ")
    # print(f"Val_len {len(val_loader)*config.batch_size} samples, ")
    print(f"Test_len {len(test_loader)*config.batch_size} samples\n")



    if config.pre_trained:
        
        model.load_state_dict(torch.load(config.best_model)['model_state_dict'])
        # Test

        test_loss,avg_iter_time, test_avg_location_eror,test_avg_euc_end_effector_eror,test_max_euc_end_effector_eror = test_model(
            model=model,
            config=config,
            data_loader=test_loader,
            data_processor=data_processor,
            device=device,
        )
        

        plot_results(config=config,data_loader=test_loader,model=model,data_processor=data_processor,device=device)
    else:
        
        best_model_checkpoint_path ,best_model_checkpoint= train(config=config, train_loader=train_loader,
                    val_loader=test_loader, 
                    model=model,
                    data_processor=data_processor, 
                    device=device, 
                    wandb_run=run)
        

        model.load_state_dict(torch.load(best_model_checkpoint_path)['model_state_dict'])

        # Test

        test_loss,avg_iter_time, test_avg_location_eror,test_avg_euc_end_effector_eror,test_max_euc_end_effector_eror = test_model(
            model=model,
            config=config,
            data_loader=test_loader,
            data_processor=data_processor,
            device=device,
            make_pdf=config.wandb_on,
            task='test',
        )
        

        plot_results(config=config,data_loader=test_loader,model=model,data_processor=data_processor,device=device)

            # print("eror in plot")
    print(f'Test_Loss: {test_loss} \n Test_Avarege_Location_Eror:{test_avg_location_eror} \n Test_Max_Euclidian_End_Effector_Eror : {test_max_euc_end_effector_eror} \n Test_Avarege_Euclidian_End_Effector_Eror: {test_avg_euc_end_effector_eror}')
    if config.wandb_on:
        
        artifact = wandb.Artifact('model', type='model')
        artifact.add_file(best_model_checkpoint_path)
        wandb.log_artifact(artifact)
        wandb.log({"Test_Loss": test_loss,"Test_Avarege_Location_Eror":test_avg_location_eror ,"Test_Max_Euclidian_End_Effector_Eror" : test_max_euc_end_effector_eror , "Test_Avarege_Euclidian_End_Effector_Eror": test_avg_euc_end_effector_eror})
        wandb.finish()

    


if __name__ == '__main__':

    # torch.autograd.set_detect_anomaly(True)

    torch.cuda.empty_cache()
    main()
