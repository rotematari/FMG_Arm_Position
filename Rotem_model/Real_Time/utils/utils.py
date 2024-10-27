import yaml
import torch
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
import serial

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration YAML file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def initialize_device(config: dict) -> torch.device:
    """
    Initialize the computation device (CPU or GPU).

    Args:
        config (dict): Configuration dictionary.

    Returns:
        torch.device: The initialized computation device.
    """
    if config["device"] == "cuda" and torch.cuda.is_available():
        logger.info("CUDA is available. Using GPU: %s", torch.cuda.get_device_name(0))
        return torch.device("cuda:0")
    else:
        logger.info("CUDA is not available or CPU is selected. Using CPU.")
        return torch.device("cpu")

def initialize_model(model_class, config: dict, device: torch.device) -> torch.nn.Module:
    """
    Initialize and return the model.

    Args:
        model_class: The model class to be instantiated.
        config (dict): Configuration dictionary.
        device (torch.device): The computation device.

    Returns:
        torch.nn.Module: The initialized model.
    """
    return model_class(config).to(device)

def initialize_scaler(checkpoint: dict) -> tuple:
    """
    Initialize and return the feature and label scalers.

    Args:
        checkpoint (dict): Model checkpoint dictionary.

    Returns:
        tuple: A tuple containing the feature and label scalers.
    """
    label_scaler = StandardScaler()
    feature_scaler = StandardScaler()
    std_feature_scaler_state = checkpoint['std_feature_scaler_state']
    std_label_scaler_state = checkpoint['std_label_scaler_state']
    
    label_scaler.mean_ = np.array(std_label_scaler_state['mean'])
    label_scaler.var_ = np.array(std_label_scaler_state['var'])
    label_scaler.scale_ = np.array(std_label_scaler_state['scale'])
    label_scaler.n_samples_seen_ = std_label_scaler_state['n_samples_seen']
    
    feature_scaler.mean_ = np.array(std_feature_scaler_state['mean'])
    feature_scaler.var_ = np.array(std_feature_scaler_state['var'])
    feature_scaler.scale_ = np.array(std_feature_scaler_state['scale'])
    feature_scaler.n_samples_seen_ = std_feature_scaler_state['n_samples_seen']
    
    return feature_scaler, label_scaler

def create_sliding_sequences(input_array: np.ndarray, sequence_length: int) -> np.ndarray:
    """
    Create sliding sequences from the input array.

    Args:
        input_array (numpy.ndarray): Input data array.
        sequence_length (int): Length of each sequence.

    Returns:
        numpy.ndarray: Array of sequences.
    """
    sample_size, features = input_array.shape
    new_sample_size = sample_size - sequence_length + 1

    sequences = [input_array[i:i+sequence_length] for i in range(new_sample_size)]
    return np.array(sequences)

def readline(serial_connection: serial.Serial) -> list:
    """
    Read a line of data from the serial connection.

    Args:
        serial_connection (serial.Serial): The serial connection object.

    Returns:
        list: A list of data points.
    """
    try:
        serial_connection.reset_input_buffer()
        line = serial_connection.readline().decode("utf-8").rstrip(',\r\n').split(',')
        
        if len(line) == 32:
            return [int(num) for num in line]
        else:
            logger.warning("Bad reading: Expected 32 data points, got %d", len(line))
            return []
    except Exception as e:
        logger.error('Error reading line: %s', e)
        return []

def load_checkpoint(checkpoint_path: str) -> dict:
    """
    Load the model checkpoint.

    Args:
        checkpoint_path (str): Path to the model checkpoint file.

    Returns:
        dict: Loaded checkpoint dictionary.
    """
    return torch.load(checkpoint_path)