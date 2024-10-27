import torch
from torch import Tensor
import pandas as pd
from os import listdir
from os.path import join
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os 
from sklearn.preprocessing import MinMaxScaler,StandardScaler
# # Change the current working directory to the directory of the main script
# os.chdir(join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
import time
from sklearn.model_selection import train_test_split
from utils.utils import center_diff_torch, subtract_bias, create_sliding_sequences,center_diff,create_sliding_sequences_old
<<<<<<< HEAD
from sklearn.decomposition import PCA
=======

>>>>>>> 1276685 (26/08)

class DataProcessor:
    def __init__(self, config ):
        self.config = config
        self.data = None
        self.label_max_val = None
        self.label_min_val = None
        self.fmg_max_val = None
        self.fmg_min_val = None
        self.label_size = config.num_labels

        if config.with_velocity:
            self.label_index = config.label_index + config.velocity_label_inedx
            self.label_size = len(self.label_index)
        else:
            self.label_index = config.label_index
            self.label_size = len(self.label_index)

        if config.norm == 'minmax':
            self.feature_scaler = MinMaxScaler(feature_range=(-1,1))
            self.label_scaler = MinMaxScaler(feature_range=(-1,1))
        elif config.norm == 'std':
            if self.config.subtract_bias:
                self.feature_scaler = StandardScaler(with_mean=False)
            else:
                self.feature_scaler = StandardScaler(with_mean=True)
            self.label_scaler =StandardScaler()

    def load_data(self) -> Tensor:
        config = self.config
        #train
        self.train_data = self.data_loader(config,config.train_data_path)
        #test
        self.test_data = self.data_loader(config,config.test_data_path)

        

    def preprocess_data(self):
        def preprocess(data):
            # average rolling window
            # data[self.config.fmg_index] = data[self.config.fmg_index].rolling(window=self.config.window_size).mean()
<<<<<<< HEAD
            # data[self.config.fmg_index] = data[self.config.fmg_index].ewm(span=self.config.window_size).mean()
            # data[self.config.label_index] = data[self.config.label_index].ewm(span=self.config.window_size).mean()
            # data[self.config.fmg_index] = self.numpy_ewma_matrix(data[self.config.fmg_index].to_numpy(), self.config.window_size)
=======
            data[self.config.fmg_index] = data[self.config.fmg_index].ewm(span=self.config.window_size).mean()
            # data[self.config.label_index] = data[self.config.label_index].ewm(span=self.config.window_size).mean()
>>>>>>> 1276685 (26/08)

            if self.config.with_velocity:
                start = time.time()
                data = center_diff_torch(self.config, locations=data,order=4)
                end = time.time()
                print(f"time it took to calc velocity {end - start} sec")
                # data[self.config.dfmg_index + self.config.velocity_label_inedx] = data[self.config.dfmg_index + self.config.velocity_label_inedx].ewm(span=100).mean()
                data[self.config.velocity_label_inedx] = data[self.config.velocity_label_inedx].ewm(span=100).mean()
                time_stamp = time.strftime("%d_%m_%H_%M", time.gmtime())
                file_name = time_stamp + '_full_data_with_velocity.csv'
                data.to_csv(file_name)
            # scale label to milimiters 
            data[self.label_index] *= 100
            
            # normalize
            self.feature_scaler.fit(data[self.config.fmg_index])
            data[self.config.fmg_index] = self.feature_scaler.transform(data[self.config.fmg_index])
            if self.config.norm_labels:
                if self.config.velocity_model:
                    self.label_scaler.fit(data[self.label_index +self.config.velocity_label_inedx])
                    data[self.label_index + self.config.velocity_label_inedx] = self.label_scaler.transform(data[self.label_index +self.config.velocity_label_inedx])
                else:
                    self.label_scaler.fit(data[self.label_index])
                    data[self.label_index] = self.label_scaler.transform(data[self.label_index])
<<<<<<< HEAD
            
            
=======
>>>>>>> 1276685 (26/08)
            data = data.drop_duplicates().dropna().reset_index(drop=True)
            return data
        
        self.train_data = preprocess(self.train_data)
        self.test_data = preprocess(self.test_data)

    def get_data_loaders(self):
        if self.train_data is None:
            print("Data not loaded. Please load the data first.")
            return None, None

        # Creating sequences
        train_features = create_sliding_sequences(torch.tensor(self.train_data[self.config.fmg_index].to_numpy(), dtype=torch.float32), self.config.sequence_length)

        # Creating sequences
        if self.config.velocity_model:
            train_labels = create_sliding_sequences(torch.tensor(self.train_data[self.label_index+ self.config.velocity_label_inedx].to_numpy(), dtype=torch.float32), self.config.sequence_length)
            test_labels = create_sliding_sequences(torch.tensor(self.test_data[self.label_index + self.config.velocity_label_inedx].to_numpy(), dtype=torch.float32), self.config.sequence_length)
        else:
            train_labels = create_sliding_sequences(torch.tensor(self.train_data[self.label_index].to_numpy(), dtype=torch.float32), self.config.sequence_length)
            test_labels = create_sliding_sequences(torch.tensor(self.test_data[self.label_index].to_numpy(), dtype=torch.float32), self.config.sequence_length)

        test_features = create_sliding_sequences(torch.tensor(self.test_data[self.config.fmg_index].to_numpy(), dtype=torch.float32), self.config.sequence_length)
        
        train_dataset = TensorDataset(torch.tensor(train_features[:train_labels.shape[0],:,:], dtype=torch.float32), torch.tensor(train_labels, dtype=torch.float32),
                                    torch.tensor(train_features[:train_labels.shape[0],:,-1:], dtype=torch.float32))

        test_dataset = TensorDataset(torch.tensor(test_features[:test_labels.shape[0],:,:], dtype=torch.float32), torch.tensor(test_labels, dtype=torch.float32),
                                    torch.tensor(test_features[:test_labels.shape[0],:,-1:], dtype=torch.float32))
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=self.config.shuffle_train, drop_last=True)

        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False, drop_last=True)

        return train_loader,test_loader

    def plot(self, from_indx=2000, to_indx=1000):
        fmg_df = self.train_data[self.config.fmg_index]
        label_position = self.train_data[self.config.label_index]

        if self.config.with_velocity:
            label_velocity = self.train_data[self.config.velocity_label_index]

        # Create a figure and a grid of subplots with 1 row and 2 columns
        fig, axes = plt.subplots(2, 1, sharex=True, sharey=False)  # Adjust figsize as needed

        # Plot data on the second subplot
        axes[0].plot(fmg_df[from_indx:to_indx])
        axes[0].set_title('Plot of FMG')
        axes[0].set_ylim([-10,10])

        axes[1].plot(label_position[from_indx:to_indx])
        axes[1].set_title('Plot of label_position')
        axes[1].set_ylim([-5,5])


        if self.config.with_velocity:
            axes[2].plot(label_velocity[from_indx:to_indx])
            axes[2].set_title('Plot of label_velocity')


        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Show the plots
        plt.show()
        return
    def data_loader(self,config,data_path):
        """
        Given a directory path, loads all the csv files in the directory and returns a concatenated pandas dataframe.
        """
        
        full_df = pd.DataFrame()
        for file in listdir(data_path):
            df = pd.read_csv(join(data_path,file))
            full_df = pd.concat([full_df,df],axis=0,ignore_index=True)
            full_df = full_df.replace(-np.inf, np.nan)
            full_df = full_df.replace(np.inf, np.nan)

        if self.config.velocity_model:
            idx = config.fmg_index + config.label_index + config.velocity_label_inedx + config.session_time_stamp 
        else:
            idx = config.fmg_index + config.label_index + config.session_time_stamp 

        return full_df[idx]
    
    def custom_train_test_split(self,dataset, test_size, test_batch_size):

        # Calculate the number of batches in the test set
        num_test_batches = test_size // test_batch_size

        # Calculate the sampling interval
        sample_interval = len(dataset) // num_test_batches

        # Create an array to store test indices
        test_indices = np.array([], dtype=int)

        # Sample test indices
        for i in range(num_test_batches):
            start_idx = i * sample_interval
            end_idx = start_idx + test_batch_size
            test_indices = np.concatenate((test_indices, np.arange(start_idx, end_idx)))

        # Ensure that test indices do not exceed the dataset length
        test_indices = test_indices[test_indices < len(dataset)]

        # Split the dataset
        test_set = dataset[test_indices]
        train_set = np.delete(dataset, test_indices,axis=0)

        return train_set, test_set
    
    def create_sliding_sequences(self,input_array, sequence_length):

        sample_size, features = input_array.shape
        new_sample_size = sample_size - sequence_length + 1

        # Preallocate the array for sequences
        sequences = np.empty((new_sample_size, sequence_length, features), dtype=input_array.dtype)

        for i in range(new_sample_size):
            sequences[i] = input_array[i:i + sequence_length]

        return sequences
    
    def numpy_ewma(self,data, window):

        alpha = 2 / (window + 1)
        alpha_rev = 1 - alpha
        n = data.shape[0]

        pows = alpha_rev**(np.arange(n+1)) + 1e-30
        
        scale_arr = 1/pows[:-1]
        offset = data*pows[1:]
        pw0 = alpha*alpha_rev**(n-1)
        
        mult = data*pw0*scale_arr
        cumsums = mult.cumsum()
        out = offset + cumsums*scale_arr[::-1]
        
        return out
    
    def numpy_ewma_matrix(self,data, window):

        for sensor in range(data.shape[1]):
            data[:,sensor] = self.numpy_ewma(data[:,sensor], window)
        
        return data