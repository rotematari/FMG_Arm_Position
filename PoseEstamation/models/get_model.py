from models.models import Conv2DLSTMAttentionModel,TransformerModel,DLinear,iTransformerModel,iTransformer2DModel,FullyConnectedNetwork,CNN1D,CNNLSTMModel
import torch 
def get_model(config):

    if config["model"] == 'Conv2DLSTMAttentionModel':
        model = Conv2DLSTMAttentionModel(config)
    elif config["model"] == 'TransformerModel':
        model = TransformerModel(config)
    elif config["model"] == 'iTransformerModel':
        model = iTransformerModel(config)
    elif config["model"] == 'iTransformer2DModel':
        model = iTransformer2DModel(config)
    elif config["model"] == 'FullyConnectedNetwork':
        model = FullyConnectedNetwork(config)
    elif config["model"] == 'CNN1D':
        model = CNN1D(config)
    elif config["model"] == 'DLinear':
        model = DLinear(config)
    elif config["model"] == 'CNNLSTMModel':
        model = CNNLSTMModel(config)

    # Load the pre-trained model if specified
    if config.get("pre_trained", False):  # Check if pre_trained key exists and is True
        checkpoint_path = config["pre_trained_path"]  # Path to the checkpoint
        print(f"Loading pre-trained model from {checkpoint_path}")
        
        # Load the checkpoint (device handling is done later)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')  # Always load to CPU first
        model.load_state_dict(checkpoint['model_state_dict'])  # Load the model weights


    return model