import pandas as pd 
import matplotlib.pyplot as plt
import argparse
import numpy as np

import yaml
import os 
# Get the current directory of the script being run
current_directory = os.path.dirname(os.path.realpath(__file__))

# Navigate up  directori
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))

# Change the working directory
os.chdir(parent_directory)
# from utils.utils import mask


with open(r'config.yaml', 'r') as f:
    args = yaml.safe_load(f)

config = argparse.Namespace(**args)

def mask(data,config):

    # create a mask that selects rows where the values in fmg_index columns are greater than 1024
    mask1 = (data[config.fmg_index] > 550).any(axis=1)

    # create a mask that selects rows where the values in first_position_label_index columns are greater than 2
    mask2 = (data[config.label_index] > 4).any(axis=1)

    # combine the masks using the | (or) operator
    mask = mask1 | mask2

    # drop the rows from the DataFrame
    data = data.drop(data[mask].index)


    return data
def plot_data(config,data):


    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(40,5))
    position = data[config.label_index]

    ax1.plot(position)
    ax2.plot(data[config.fmg_index])
    ax1.legend()

    plt.show() 

1
def is_not_numeric(x):
    try:
        float(x)
        return False
    except ValueError:
        return True
    

def print_not_numeric_vals(df):

    mask = df[config.fmg_index+config.label_index].applymap(is_not_numeric)
    non_numeric_values = df[mask].stack().dropna()
    print(non_numeric_values)

    return non_numeric_values    

def clean_data(df,not_numeric_vals):

    x = not_numeric_vals.index.to_numpy()
    index_list = [x[i][0] for i in range(len(x)) ]
    df = df.drop(index_list).reset_index(drop=True)
    df[config.fmg_index+config.label_index]=df[config.fmg_index+config.label_index].astype(np.float64)

    return df


if __name__== '__main__':

    # df = pd.read_csv(r'./data/data/11_Sep_2023_14_50_movment_2.csv')
    # df = df[config.fmg_index+config.first_positoin_label_inedx+config.sesion_time_stamp] 

    # not_numeric_vals = print_not_numeric_vals(df)

    # if not_numeric_vals.shape[0] == 0:
    #     plot_data(config=config,data=df)
    #     clean_df=df

    # else :
    #     print("clean data")
    #     clean_df = clean_data(df,not_numeric_vals)
    #     clean_df = mask(clean_df,config=config)
    #     plot_data(config=config,data=clean_df)


    # x = input("to save press 1\n ")

    # if x == '1' :
    #     clean_df.to_csv(r'./data/data/11_Sep_2023_14_50_movment_2_clean.csv',index=False)

        # Directory containing the CSV files
    directory = './data/toclean'

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            # Construct the full file path
            file_path = os.path.join(directory, filename)

            # Read the CSV file
            df = pd.read_csv(file_path)
            df = df[config.fmg_index + config.label_index + config.session_time_stamp]
            
            

            not_numeric_vals = print_not_numeric_vals(df)


            if not_numeric_vals.shape[0] == 0:
                print("data is clean")
                plot_data(config=config, data=df)
                x = input(f"To mask data {filename}, press 1\n")
                if x == '1':
                    clean_df = mask(df, config=config)
                
            else:
                print("clean data")
                clean_df = clean_data(df, not_numeric_vals)
                clean_df = mask(clean_df, config=config)
            plot_data(config=config, data=clean_df)
            
            # Prompt to save the cleaned data
            x = input(f"To save cleaned data for {filename}, press 1\n")

            if x == '1':
                # Save the cleaned data to a new CSV file
                clean_file_path = os.path.join(directory, filename.split('.')[0] + '_clean.csv')
                clean_df.to_csv(clean_file_path, index=False)