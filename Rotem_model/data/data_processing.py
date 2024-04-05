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

from sklearn.model_selection import train_test_split
from utils.utils import center_diff, subtract_bias, create_sliding_sequences

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

    def load_data(self,train = True) -> Tensor:
        config = self.config
        #train
        config.data_path = config.train_data_path
        self.train_data = self.data_loader(config)
        #test
        config.data_path = config.test_data_path
        self.test_data = self.data_loader(config)

        

    def preprocess_data(self):
        #pre process train 
        if not self.config.data_path == './data/FullData':
            if self.config.with_velocity:
                # adds velocities to the labels
                self.train_data = center_diff(self.config, locations=self.train_data,order=4)
            # subtracts the bias on the FMG sensors data
            if self.config.subtract_bias:
                self.train_data[self.config.fmg_index] = subtract_bias(self.train_data[self.config.fmg_index + self.config.session_time_stamp])
                self.train_data = self.train_data.drop_duplicates().dropna().reset_index(drop=True)

        # average rolling window
        if self.config.with_velocity:
            self.train_data[self.config.fmg_index + self.config.velocity_label_inedx] = self.train_data[self.config.fmg_index+ self.config.velocity_label_inedx].rolling(window=self.config.window_size).mean()
        else:
            # self.train_data[self.config.fmg_index] = self.train_data[self.config.fmg_index].rolling(window=self.config.window_size).mean()
            self.train_data[self.config.fmg_index] = self.train_data[self.config.fmg_index].ewm(span=self.config.window_size).mean()

        # normalize
        self.feature_scaler.fit(self.train_data[self.config.fmg_index])
        self.train_data[self.config.fmg_index] = self.feature_scaler.transform(self.train_data[self.config.fmg_index])
        
        # for i, var in enumerate(self.feature_scaler.var_):
        #     if var < 50:
        #         self.train_data.iloc[:,i] = 0

        if self.config.norm_labels:
            self.label_scaler.fit(self.train_data[self.label_index])
            self.train_data[self.label_index] = self.label_scaler.transform(self.train_data[self.label_index])
        self.train_data = self.train_data.drop_duplicates().dropna().reset_index(drop=True)
        #pre process test
        # subtracts the bias on the FMG sensors data
        if self.config.subtract_bias:
            self.test_data[self.config.fmg_index] = subtract_bias(self.test_data[self.config.fmg_index + self.config.session_time_stamp])
            self.test_data = self.test_data.drop_duplicates().dropna().reset_index(drop=True)

        # average rolling window
        if self.config.with_velocity:
            self.test_data[self.config.fmg_index + self.config.velocity_label_inedx] = self.test_data[self.config.fmg_index+ self.config.velocity_label_inedx].rolling(window=self.config.window_size).mean()
        else:
            # self.test_data[self.config.fmg_index] = self.test_data[self.config.fmg_index].rolling(window=self.config.window_size).mean()
            self.test_data[self.config.fmg_index] = self.test_data[self.config.fmg_index].ewm(span=self.config.window_size).mean()
        
        # normalize
        self.test_data[self.config.fmg_index] = self.feature_scaler.transform(self.test_data[self.config.fmg_index])
        if self.config.norm_labels:
            self.test_data[self.label_index] = self.label_scaler.transform(self.test_data[self.label_index])
        
        self.test_data = self.test_data.drop_duplicates().dropna().reset_index(drop=True)

    def get_data_loaders(self):
        if self.train_data is None:
            print("Data not loaded. Please load the data first.")
            return None, None

        train_fmg_df = self.train_data[self.config.fmg_index]
        train_label_df = self.train_data[self.label_index]
        # time_feature = self.data[self.config.time_stamp]


        # Creating sequences
        train_features = create_sliding_sequences(torch.tensor(train_fmg_df.to_numpy(), dtype=torch.float32), self.config.sequence_length)
        train_labels = create_sliding_sequences(torch.tensor(train_label_df.to_numpy(), dtype=torch.float32), self.config.sequence_length)

        #test 
        test_fmg_df = self.test_data[self.config.fmg_index]
        test_label_df = self.test_data[self.label_index]
        # time_feature = self.data[self.config.time_stamp]


        # Creating sequences
        test_features = create_sliding_sequences(torch.tensor(test_fmg_df.to_numpy(), dtype=torch.float32), self.config.sequence_length)
        test_labels = create_sliding_sequences(torch.tensor(test_label_df.to_numpy(), dtype=torch.float32), self.config.sequence_length)

        # features = self.create_sliding_sequences(fmg_df.to_numpy(), self.config.sequence_length)
        # labels = self.create_sliding_sequences(label_df.to_numpy(), self.config.sequence_length)
        # time_feature =  create_sliding_sequences(torch.tensor(time_feature.to_numpy(), dtype=torch.float32), self.config.sequence_length)
            
        # features = torch.cat((features, time_feature), dim=2)
        # labels = torch.cat((labels, time_feature), dim=2)



        # Split the data into training and test sets
        # shape 
        # train_fmg, test_fmg, train_label, test_label = train_test_split(features, labels, test_size=self.config.test_size, random_state=self.config.random_state, shuffle=self.config.shuffle)
        # Split the training data into training and validation sets
        # train_fmg, val_fmg, train_label, val_label = train_test_split(train_fmg, train_label, test_size=self.config.val_size / (1 - self.config.test_size), random_state=self.config.random_state, shuffle=False)
        # train_fmg, test_fmg = self.custom_train_test_split(features,self.config.test_size,self.config.test_batch_size)
        # train_label, test_label = self.custom_train_test_split(labels,self.config.test_size,self.config.test_batch_size)
        
        # train_dataset = TensorDataset(train_fmg[:train_label.shape[0],:,:], train_label,
        #                               train_fmg[:train_label.shape[0],:,-1:])
        # # val_dataset = TensorDataset(val_fmg[:val_label.shape[0],:,:-1], val_label,val_fmg[:val_label.shape[0],:,-1:])
        # test_dataset = TensorDataset(test_fmg[:test_label.shape[0],:,:], test_label,
        #                             test_fmg[:test_label.shape[0],:,-1:])
        # train_dataset = TensorDataset(torch.tensor(train_fmg[:train_label.shape[0],:,:], dtype=torch.float32), torch.tensor(train_label, dtype=torch.float32),
        #                             torch.tensor(train_fmg[:train_label.shape[0],:,-1:], dtype=torch.float32))
        # # val_dataset = TensorDataset(val_fmg[:val_label.shape[0],:,:-1], val_label,val_fmg[:val_label.shape[0],:,-1:])
        # test_dataset = TensorDataset(torch.tensor(test_fmg[:test_label.shape[0],:,:], dtype=torch.float32), torch.tensor(test_label, dtype=torch.float32),
        #                             torch.tensor(test_fmg[:test_label.shape[0],:,-1:], dtype=torch.float32))
        
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
    def data_loader(self,config):
        """
        Given a directory path, loads all the csv files in the directory and returns a concatenated pandas dataframe.
        """
        
        full_df = pd.DataFrame()
        for file in listdir(config.data_path):
            df = pd.read_csv(join(config.data_path,file))
            full_df = pd.concat([full_df,df],axis=0,ignore_index=True)
            full_df = full_df.replace(-np.inf, np.nan)
            full_df = full_df.replace(np.inf, np.nan)
        if config.with_velocity:
            idx = config.fmg_index + config.label_index +config.velocity_label_inedx + config.session_time_stamp # +config.time_stamp
        else:
            idx = config.fmg_index + config.label_index + config.session_time_stamp #+ config.time_stamp

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