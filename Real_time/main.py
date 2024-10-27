import yaml
import torch
from sklearn.preprocessing import StandardScaler
from models.transformer_model import TransformerModel
from real_time.data_reader import DataReader
from real_time.prediction import Prediction
from real_time.plot import DynamicPlot
from real_time.natnet_handler import NatNetReader
from utils.logger import setup_logger

def load_checkpoint(config):
    checkpoint = torch.load(config['real_time']['checkpoint_path'])
    config.update(checkpoint['config'])  # Update config with saved settings
    return checkpoint

def initialize_model(config, checkpoint):
    model = TransformerModel(config).to(torch.device(config['real_time']['device']))
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def initialize_scalers(checkpoint):
    label_scaler = StandardScaler()
    feature_scaler = StandardScaler()
    
    std_feature_scaler_state = checkpoint['feature_scaler_state']
    std_label_scaler_state = checkpoint['label_scaler_state']

    label_scaler.mean_ = std_label_scaler_state['mean']
    label_scaler.var_ = std_label_scaler_state['var']
    label_scaler.scale_ = std_label_scaler_state['scale']
    label_scaler.n_samples_seen_ = std_label_scaler_state['n_samples_seen']

    feature_scaler.mean_ = std_feature_scaler_state['mean']
    feature_scaler.var_ = std_feature_scaler_state['var']
    feature_scaler.scale_ = std_feature_scaler_state['scale']
    feature_scaler.n_samples_seen_ = std_feature_scaler_state['n_samples_seen']
    
    return feature_scaler, label_scaler

if __name__ == "__main__":
    # Load config
    with open("config/real_time_config.yaml", 'r') as file:
        config = yaml.safe_load(file)
    
    logger = setup_logger()

    # Load checkpoint
    checkpoint = load_checkpoint(config)

    # Initialize model
    model = initialize_model(config, checkpoint)

    # Initialize scalers
    feature_scaler, label_scaler = initialize_scalers(checkpoint)
    
    # Initialize components
    data_reader = DataReader(config)
    plot = DynamicPlot()
    natnet_reader = NatNetReader(config['real_time'])
    natnet_reader.connect()

    # Prediction
    prediction = Prediction(model, feature_scaler, label_scaler, config['real_time'])

    try:
        while True:
            sequence = data_reader.get_sequence()
            processed_sequence = feature_scaler.transform(sequence)
            pred = prediction.predict(processed_sequence)
            # print(pred)
            if pred is not None:
                ground_truth = natnet_reader.read_sample()
                plot.update_plot(pred, ground_truth)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        natnet_reader.disconnect()