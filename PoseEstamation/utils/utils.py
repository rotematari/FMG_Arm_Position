import torch
import random
import numpy as np
import os
import yaml
import pandas as pd

def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Running on the CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Running on CPU")
    return device

def set_seed(seed, torch_deterministic=False, rank=0):
    """ set seed across modules """
    if seed == -1 and torch_deterministic:
        seed = 42 + rank
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    else:
        seed = seed + rank

    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # added these to test if it helps with reproducibility
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed
def load_yaml_config(file_path):
    """
    Loads a YAML configuration file and returns the contents as a dictionary.
    
    Parameters:
    file_path (str): Path to the YAML configuration file.
    
    Returns:
    dict: Contents of the YAML file as a dictionary.
    """
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
            return config
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse YAML file. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def save_to_csv(data: np.ndarray, model_name: str = None):
    # Define the column labels
    columns = ['Ex', 'Ey', 'Ez', 'Wx', 'Wy', 'Wz']
    
    # Create a DataFrame from the numpy array
    df = pd.DataFrame(data, columns=columns)
    # Create directory path
    save_dir = "results/saved_predictions"
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
    
    # Define the filename
    if model_name:
        filename = f"{model_name}_preds.csv"
    else:
        filename = "targets.csv"
    
    # Save the DataFrame to CSV
    # Save the DataFrame to CSV
    file_path = os.path.join(save_dir, filename)
    df.to_csv(file_path, index=False)
    print(f"Data saved to {filename}")