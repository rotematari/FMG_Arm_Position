import time
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from data_processes.data_loader import FMGPoseDataset
from utils.utils import load_yaml_config, set_device, set_seed
from models.get_model import get_model

if __name__ == '__main__':
    # Load the config file
    config = load_yaml_config('config.yaml')
    # Set the seed
    set_seed(config["seed"])
    device = set_device()

    # Load the dataset
    trainDataset = FMGPoseDataset(config, mode='train')
    testDataset = FMGPoseDataset(config, mode='test', feature_scalar=trainDataset.feature_scalar, label_scalar=trainDataset.label_scalar)
    print(f"Train Data set length {len(trainDataset)}")
    print(f"Test Data set length {len(testDataset)}")

    # Create the dataloader
    trainDataloader = DataLoader(trainDataset, batch_size=1, shuffle=True, drop_last=True)
    testDataloader = DataLoader(testDataset, batch_size=1, shuffle=False, drop_last=True)

    # Create the model
    model = get_model(config=config)
    model = model.to(device=device)

    # Create the optimizer
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    # Measure inference time
    model.eval()  # Set the model to evaluation mode
    i = 0
    # Select a batch from the test data loader
    with torch.no_grad():  # No need for gradient calculation during inference
        inference_times = []
        for batch in testDataloader:
            inputs, labels = batch
            inputs = inputs.to(device)

            # Start timing
            start_time = time.time()

            # Forward pass (inference)
            outputs = model(inputs)

            # End timing
            end_time = time.time()

            # Calculate inference time for the current batch
            inference_time = end_time - start_time
            inference_times.append(inference_time)

            print(f"Inference time for batch: {inference_time:.6f} seconds")
            i+=1
            if i>=1000:
                break
    avg_inference_time = sum(inference_times) / len(inference_times)
    print(f"Average inference time: {avg_inference_time:.6f} seconds")
