from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from os import listdir
from os.path import join
from sklearn.preprocessing import StandardScaler
import os
from random import sample
class FMGPoseDataset(Dataset):
    def __init__(self,config,mode,feature_scalar = None,label_scalar = None):
        
        self.config = config
        self.sequence_length = config["sequence_length"]
        self.mode = mode
        self.data = self.load_data()
        if mode == "train":
            self.feature_scalar = StandardScaler()
            self.label_scalar = StandardScaler()
        elif mode == "test":
            self.feature_scalar = feature_scalar
            self.label_scalar = label_scalar
        elif mode == "fine_tune":
            self.feature_scalar = feature_scalar
            self.label_scalar = label_scalar

        self.data = self.preprocess()

        self.data = self.stride_Sampling()

    def __len__(self):
        length = (len(self.data)- self.sequence_length)
        if self.mode == "train" :
            length = (len(self.data)- self.sequence_length)//(self.sequence_length//2)
            
        # length = (len(self.data)- self.sequence_length)//(self.sequence_length//2)
        return length 
    
    def __getitem__(self,idx):
        sequence_length = self.config["sequence_length"]
        feature_index_size = len(self.config["feature_index"])
        
        if self.mode == "train" :
            idx = idx*sequence_length//2

        # Ensure the index does not go beyond the dataset
        if idx > len(self.data):
            raise IndexError("Index out of bounds for the dataset")
        data = self.data
        
        inputs = torch.tensor(data[idx : idx + sequence_length,:feature_index_size],dtype=torch.float32)
        targets = torch.tensor(data[idx + sequence_length,feature_index_size:],dtype=torch.float32)
        return inputs , targets
    
    def load_data(self):
        """
        Given a directory path, loads all the csv files in the directory and returns a concatenated pandas dataframe.
        """
        if self.mode == "train":
            data_path = self.config["train_data_path"]
        elif self.mode == "fine_tune":
            data_path = self.config["fine_tuning"]["train_data_path"]
        elif self.mode == "test_fine_tune":
            data_path = self.config["fine_tuning"]["test_data_path"]
        else:
            data_path = self.config["test_data_path"]

        
        feature_index = self.config["feature_index"]
        label_index = self.config["label_index"]


        idx =feature_index + label_index
        
        
        df_list = []
        for file in listdir(data_path):
            df = pd.read_csv(join(data_path, file))
            df = df.replace([np.inf, -np.inf], np.nan)
            df_list.append(df)

        full_df = pd.concat(df_list, axis=0, ignore_index=True)
        full_df = full_df.dropna().reset_index(drop=True)

        return full_df[idx].to_numpy()

    def preprocess(self):
        """
        Preprocess the data by normalizing it and splitting it into sequences.
        """

        data = self.data
        

        feature_index_size = len(self.config["feature_index"])

        normelized_fetures= self.feature_scalar.fit_transform(data[:,:feature_index_size])
        normelized_labels = self.label_scalar.fit_transform(data[:,feature_index_size:])

        data = np.concatenate((normelized_fetures,normelized_labels),axis=1)
        return data

    def stride_Sampling(self):
        data = self.data
        data = self.data
        config = self.config
        sample_size, features = data.shape
        jump = config["jump"]
        sampled_array = []
        j = 0
        while j <=  sample_size - jump:
            sampled_array.append(data[j])
            j += jump
        return np.array(sampled_array)


class FMGPoseDataset_data_importance (Dataset):
    def __init__(self,config,mode,number_of_data_samples,feature_scalar = None,label_scalar = None):
        
        self.config = config
        self.sequence_length = config["sequence_length"]
        self.mode = mode
        self.data = self.load_random_data(number_of_data_samples)
        if mode == "train":
            self.feature_scalar = StandardScaler()
            self.label_scalar = StandardScaler()
        elif mode == "test":
            self.feature_scalar = feature_scalar
            self.label_scalar = label_scalar

        self.data = self.preprocess()

        self.data = self.stride_Sampling()

    def __len__(self):
        length = (len(self.data)- self.sequence_length)
        if self.mode == "train":
            length = (len(self.data)- self.sequence_length)//(self.sequence_length//2)
            
        
        return length 
    
    def __getitem__(self,idx):
        sequence_length = self.config["sequence_length"]
        feature_index_size = len(self.config["feature_index"])
        
        if self.mode == "train":
            idx = idx*sequence_length//2

        # Ensure the index does not go beyond the dataset
        if idx > len(self.data):
            raise IndexError("Index out of bounds for the dataset")
        data = self.data
        
        inputs = torch.tensor(data[idx : idx + sequence_length,:feature_index_size],dtype=torch.float32)
        targets = torch.tensor(data[idx + sequence_length,feature_index_size:],dtype=torch.float32)
        return inputs , targets
    


    def load_random_data(self, n):
        """
        Given a directory path, randomly selects n CSV files, loads them, and returns a concatenated pandas dataframe.
        """
        data_path = self.config["train_data_path"]
        feature_index = self.config["feature_index"]
        label_index = self.config["label_index"]
        
        idx = feature_index + label_index
        
        # Get all files in the directory
        all_files = [file for file in os.listdir(data_path) if file.endswith('.csv')]
        
        # Ensure n doesn't exceed the number of available files
        n = min(n, len(all_files))
        
        # Randomly select n files from the directory
        self.selected_files = sample(all_files, n)
        
        df_list = []
        
        # Load the selected files
        for file in self.selected_files:
            df = pd.read_csv(os.path.join(data_path, file))
            df = df.replace([np.inf, -np.inf], np.nan)
            df_list.append(df)

        # Concatenate all dataframes
        full_df = pd.concat(df_list, axis=0, ignore_index=True)
        full_df = full_df.dropna().reset_index(drop=True)

        num_samples = full_df.shape[0]
        train_size = int(self.config["train_size"] * num_samples)

        if self.mode == 'train':
            return full_df[idx][:train_size].to_numpy()
        elif self.mode == 'test':
            return full_df[idx][train_size:].to_numpy()


    def preprocess(self):
        """
        Preprocess the data by normalizing it and splitting it into sequences.
        """

        data = self.data
        

        feature_index_size = len(self.config["feature_index"])

        normelized_fetures= self.feature_scalar.fit_transform(data[:,:feature_index_size])
        normelized_labels = self.label_scalar.fit_transform(data[:,feature_index_size:])

        data = np.concatenate((normelized_fetures,normelized_labels),axis=1)
        return data

    def stride_Sampling(self):
        data = self.data
        data = self.data
        config = self.config
        sample_size, features = data.shape
        jump = config["jump"]
        sampled_array = []
        j = 0
        while j <=  sample_size - jump:
            sampled_array.append(data[j])
            j += jump
        return np.array(sampled_array)
    

class FMGPoseDatasetFeatureImportance(Dataset):
    def __init__(self,config,mode,feature_scalar = None,label_scalar = None,sensor_to_shufle=None):

        self.sensor_to_shufle = sensor_to_shufle
        self.config = config
        self.sequence_length = config["sequence_length"]
        self.mode = mode
        self.data = self.load_data()
        if mode == "train":
            self.feature_scalar = StandardScaler()
            self.label_scalar = StandardScaler()
        elif mode == "test":
            self.feature_scalar = feature_scalar
            self.label_scalar = label_scalar

        self.data = self.preprocess()

        self.data = self.stride_Sampling()

    def __len__(self):
        length = (len(self.data)- self.sequence_length)
        if self.mode == "train" :
            length = (len(self.data)- self.sequence_length)//(self.sequence_length//2)
            
        # length = (len(self.data)- self.sequence_length)//(self.sequence_length//2)
        return length 
    
    def __getitem__(self,idx):
        sequence_length = self.config["sequence_length"]
        feature_index_size = len(self.config["feature_index"])
        
        if self.mode == "train" :
            idx = idx*sequence_length//2

        # Ensure the index does not go beyond the dataset
        if idx > len(self.data):
            raise IndexError("Index out of bounds for the dataset")
        data = self.data
        
        inputs = torch.tensor(data[idx : idx + sequence_length,:feature_index_size],dtype=torch.float32)
        targets = torch.tensor(data[idx + sequence_length,feature_index_size:],dtype=torch.float32)
        return inputs , targets
    
    def load_data(self):
        """
        Given a directory path, loads all the csv files in the directory and returns a concatenated pandas dataframe.
        """
        if self.mode == "train":
            data_path = self.config["train_data_path"]
        else:
            data_path = self.config["test_data_path"]
        
        feature_index = self.config["feature_index"]
        label_index = self.config["label_index"]


        idx =feature_index + label_index
        
        
        df_list = []
        for file in listdir(data_path):
            df = pd.read_csv(join(data_path, file))
            df = df.replace([np.inf, -np.inf], np.nan)
            df_list.append(df)

        full_df = pd.concat(df_list, axis=0, ignore_index=True)
        full_df = full_df.dropna().reset_index(drop=True)
        if self.sensor_to_shufle:
            # full_df[self.sensor_to_shufle] = full_df[self.sensor_to_shufle].sample(frac=1).reset_index(drop=True)
            full_df[self.sensor_to_shufle] = 0
        return full_df[idx].to_numpy()

    def preprocess(self):
        """
        Preprocess the data by normalizing it and splitting it into sequences.
        """

        data = self.data
        

        feature_index_size = len(self.config["feature_index"])

        normelized_fetures= self.feature_scalar.fit_transform(data[:,:feature_index_size])
        normelized_labels = self.label_scalar.fit_transform(data[:,feature_index_size:])

        data = np.concatenate((normelized_fetures,normelized_labels),axis=1)
        return data

    def stride_Sampling(self):
        data = self.data
        data = self.data
        config = self.config
        sample_size, features = data.shape
        jump = config["jump"]
        sampled_array = []
        j = 0
        while j <=  sample_size - jump:
            sampled_array.append(data[j])
            j += jump
        return np.array(sampled_array)
    

class FMGPoseDataset_important_sensors(Dataset):
    def __init__(self,config,mode,feature_scalar = None,label_scalar = None,sensor_to_use=None):
        
        self.config = config
        self.sequence_length = config["sequence_length"]
        self.sensor_to_use = sensor_to_use
        self.mode = mode
        self.data = self.load_data()
        if mode == "train":
            self.feature_scalar = StandardScaler()
            self.label_scalar = StandardScaler()
        elif mode == "test":
            self.feature_scalar = feature_scalar
            self.label_scalar = label_scalar
        elif mode == "fine_tune":
            self.feature_scalar = feature_scalar
            self.label_scalar = label_scalar

        self.data = self.preprocess()

        self.data = self.stride_Sampling()

    def __len__(self):
        length = (len(self.data)- self.sequence_length)
        if self.mode == "train" :
            length = (len(self.data)- self.sequence_length)//(self.sequence_length//2)
            
        # length = (len(self.data)- self.sequence_length)//(self.sequence_length//2)
        return length 
    
    def __getitem__(self,idx):
        sequence_length = self.config["sequence_length"]
        feature_index_size = len(self.new_feature_index)
        
        if self.mode == "train" :
            idx = idx*sequence_length//2

        # Ensure the index does not go beyond the dataset
        if idx > len(self.data):
            raise IndexError("Index out of bounds for the dataset")
        data = self.data
        
        inputs = torch.tensor(data[idx : idx + sequence_length,:feature_index_size],dtype=torch.float32)
        targets = torch.tensor(data[idx + sequence_length,feature_index_size:],dtype=torch.float32)
        return inputs , targets
    
    def load_data(self):
        """
        Given a directory path, loads all the csv files in the directory and returns a concatenated pandas dataframe.
        """
        if self.mode == "train":
            data_path = self.config["train_data_path"]
        elif self.mode == "fine_tune":
            data_path = self.config["fine_tuning"]["train_data_path"]
        elif self.mode == "test_fine_tune":
            data_path = self.config["fine_tuning"]["test_data_path"]
        else:
            data_path = self.config["test_data_path"]

        
        feature_index = self.config["feature_index"]
        label_index = self.config["label_index"]

        # build the new index list from the sensor_to_use
        self.new_feature_index = [feature_index[idx-1] for idx in self.sensor_to_use]

        idx =self.new_feature_index + label_index
        
        
        df_list = []
        for file in listdir(data_path):
            df = pd.read_csv(join(data_path, file))
            df = df.replace([np.inf, -np.inf], np.nan)
            df_list.append(df)

        full_df = pd.concat(df_list, axis=0, ignore_index=True)
        full_df = full_df.dropna().reset_index(drop=True)

        return full_df[idx].to_numpy()

    def preprocess(self):
        """
        Preprocess the data by normalizing it and splitting it into sequences.
        """

        data = self.data
        

        feature_index_size = len(self.new_feature_index)

        normelized_fetures= self.feature_scalar.fit_transform(data[:,:feature_index_size])
        normelized_labels = self.label_scalar.fit_transform(data[:,feature_index_size:])

        data = np.concatenate((normelized_fetures,normelized_labels),axis=1)
        return data

    def stride_Sampling(self):
        data = self.data
        data = self.data
        config = self.config
        sample_size, features = data.shape
        jump = config["jump"]
        sampled_array = []
        j = 0
        while j <=  sample_size - jump:
            sampled_array.append(data[j])
            j += jump
        return np.array(sampled_array)
