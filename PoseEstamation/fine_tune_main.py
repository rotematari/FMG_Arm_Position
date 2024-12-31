import torch
import logging
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_processes.data_loader import FMGPoseDataset
from utils.utils import load_yaml_config ,set_device, set_seed
from evaluation.evaluate import test_model
from models.get_model import get_model
from datetime import datetime

class FineTuner:
    def __init__(self, model, config):
        self.config = config.get('fine_tuning', {})
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)


        self.learning_rate = self.config.get('learning_rate', 1.0e-4)
        print(f'Learning rate: {self.learning_rate}')
        print(f'Weight decay: {self.config.get("weight_decay", 1.0e-5)}')
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.config.get('weight_decay', 1.0e-5)
        )

        self.load_model(self.config.get('pre_trained_model_path', 'pre_trained_model.pth'))
        self.freeze_parameters(self.config.get('layers_to_train', None))
        model_name = self.config.get('model_name', 'Model')
        timestamp = datetime.now().strftime('%d_%m_%H_%M')
        self.model_save_path = f'results/saved_models/{model_name}_epoch_{{}}_{timestamp}.pth'
        # self.model_save_path = self.config.get('model_save_path', 'fine_tuned_model.pth')

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        
        for inputs, targets in tqdm(train_loader, desc='Fine-tuning'):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            elbow_outputs, wrist_outputs, _ = self.model(inputs)
            outputs = torch.cat((elbow_outputs, wrist_outputs), dim=1)
            
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    def validate(self, val_loader,label_scaler,epoch):
        return test_model(
            model=self.model,
            config=self.config,
            data_loader=val_loader,
            label_scaler=label_scaler,
            device=self.device,
            task='validation',
            epoch=epoch)
    def fine_tune(self, train_loader, val_loader,label_scaler):
        num_epochs = self.config.get('num_epochs', 5)
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):

            train_loss = self.train_epoch(train_loader)

            val_loss,avg_critic_loss,avg_iter_time,\
            avg_location_error,avg_euc_end_effector_error,\
            max_euc_end_effector_error,R2_score,\
            avg_elbow_error,wrist_std,elbow_std = self.validate(val_loader,label_scaler,epoch)
            
            print(f'Fine-tuning Epoch {epoch+1}/{num_epochs}:')
            print(f'Training Loss: {train_loss:.4f}')
            print(f'Validation Loss: {val_loss:.4f}')
            print(f'Average Critic Loss: {avg_critic_loss:.4f}')
            print(f'Average Iteration Time: {avg_iter_time:.4f}')
            print(f'Average Location Error: {avg_location_error}')
            print(f'Average Euclidean End-Effector Error: {avg_euc_end_effector_error:.4f}')
            print(f'Maximum Euclidean End-Effector Error: {max_euc_end_effector_error:.4f}')
            print(f'R2 Score: {R2_score:.4f}')
            print(f'Average Elbow Error: {avg_elbow_error:.4f}')
            print(f'Wrist Standard Deviation: {wrist_std:.4f}')
            print(f'Elbow Standard Deviation: {elbow_std:.4f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(epoch, best_val_loss)

    def _save_checkpoint(self, epoch, loss):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, self.model_save_path)


    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['loss']
    
    def freeze_parameters(self, layers_to_train=None):
        """
        Freezes model parameters based on specified layers or all layers if none specified
        Args:
            layers_to_freeze (list): List of layer names to freeze. If None, freezes all layers
        """
        # layers_to_train = [
        #     # 'wrist_fc.2',
        #     'wrist_fc.5',
        #     'wrist_fc_sum',
        #     # 'elbow_fc.2', 
        #     'elbow_fc.5',
        #     'elbow_fc_sum'
        # ]
        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze specified layers
        if layers_to_train is not None:
            for name, param in self.model.named_parameters():
                if any(layer in name for layer in layers_to_train):
                    param.requires_grad = True

    
if __name__ == '__main__':

    # Load the config file
    config = load_yaml_config('config.yaml')
    # Set the seed
    set_seed(config["seed"])
    device = set_device()
    model = get_model(config=config)

    model = model.to(device=device) 


    # Load the dataset
    trainDataset = FMGPoseDataset(config,mode='train')
    fine_tune_dataset = FMGPoseDataset(config,mode='fine_tune',feature_scalar=trainDataset.feature_scalar,label_scalar= trainDataset.label_scalar)
    testDataset = FMGPoseDataset(config,mode='test',feature_scalar=trainDataset.feature_scalar,label_scalar= trainDataset.label_scalar)
    print(f"Train Data set length {len(fine_tune_dataset)}")
    print(f"Test Data set length {len(testDataset)}")

    # Create the dataloader
    fineTuneDataloader = DataLoader(fine_tune_dataset, batch_size=config["batch_size"], shuffle=True,drop_last=True)
    testDataloader = DataLoader(testDataset, batch_size=config["batch_size"], shuffle=False,drop_last=True)

    fine_tuner = FineTuner(model, config)
    fine_tuner.fine_tune(fineTuneDataloader, testDataloader,trainDataset.label_scalar)