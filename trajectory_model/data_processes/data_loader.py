from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from os import listdir
from os.path import join
from sklearn.preprocessing import StandardScaler

class PositionsDataset(Dataset):
    def __init__(self,config,mode):
        self.config = config
        self.sequence_length = config["sequence_length"]
        self.prediction_length = config["iTransformer_pred_length"]
        self.mode = mode
        self.data = self.load_data()
        self.scalar = StandardScaler()
        self.data = self.preprocess()
        self.data = self.stride_Sampling()
        # self.data = self.data[:10000]

    def __len__(self):
        length = (len(self.data)- self.sequence_length - self.prediction_length)//self.sequence_length
        return length 
    
    def __getitem__(self,idx):
        sequence_length = self.config["sequence_length"]
        prediction_length = self.config["iTransformer_pred_length"]
        idx = idx*sequence_length
        # Ensure the index does not go beyond the dataset
        if idx + sequence_length + prediction_length > len(self.data):
            raise IndexError("Index out of bounds for the dataset")
        data = self.data
        
        inputs = torch.tensor(data[idx : idx + sequence_length],dtype=torch.float32)
        targets = torch.tensor(data[idx + sequence_length : idx + sequence_length + prediction_length],dtype=torch.float32)
        return inputs , targets
    
    def load_data(self):
        """
        Given a directory path, loads all the csv files in the directory and returns a concatenated pandas dataframe.
        """
        data_path = self.config["data_path"]
        label_index = self.config["label_index"]

        idx =label_index
        
        full_df = pd.DataFrame()
        for file in listdir(data_path):
            df = pd.read_csv(join(data_path,file))
            full_df = pd.concat([full_df,df],axis=0,ignore_index=True)
            full_df = full_df.replace(-np.inf, np.nan)
            full_df = full_df.replace(np.inf, np.nan)
        
        full_df = full_df.dropna()
        full_df = full_df.reset_index(drop=True)

        num_samples = full_df.shape[0]
        train_size = int(self.config["train_size"]*num_samples)

        if self.mode == 'train':
            return full_df[idx][:train_size].to_numpy()
        elif self.mode == 'test':
            return full_df[idx][train_size:].to_numpy()
        
    def preprocess(self):
        """
        Preprocess the data by normalizing it and splitting it into sequences.
        """
        data = self.data
        data = self.scalar.fit_transform(data)

        return data
    def stride_Sampling(self):
        data = self.data
        config = self.config
        sample_size, features = data.shape
        jump = config["jump"]
        sampled_array = []
        j = 0
        while j <  sample_size - 1:
            sampled_array.append(data[j])
            j += jump
        return np.array(sampled_array)
    