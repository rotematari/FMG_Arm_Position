import os
import time
import yaml
import logging
import torch
import numpy as np
import pandas as pd
import serial
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from models.models import TransformerModel
from Real_Time.NatnetReader import NatNetReader
from scipy.signal import savgol_filter
from Real_Time.utils.dynamic_plot import DynamicPlot
from Real_Time.utils.utils import load_config, initialize_device, initialize_model, initialize_scaler, create_sliding_sequences, readline, load_checkpoint

# Change the current working directory to the directory of the script
# os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealTimeSystem:
    """
    A class to handle real-time data processing, prediction, and visualization using a neural network model.
    """

    def __init__(self, config_path: str):
        """
        Initialize the RealTimeSystem with a configuration file.

        Args:
            config_path (str): Path to the configuration YAML file.
        """
        self.RealTimeconfig = load_config(config_path)
        
        self.natnet_reader = NatNetReader()
        self.device = initialize_device(self.RealTimeconfig)

        self.checkpoint = load_checkpoint(self.RealTimeconfig['checkpoint_path'])
        self.config = self.checkpoint['config']
        self.model = initialize_model(TransformerModel, self.config, self.device)
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.feature_scaler, self.label_scaler = initialize_scaler(self.checkpoint)

        self.first = True
        
        if self.RealTimeconfig['testFromFile']:
            self.load_test_data()
        else:
            self.ser = self.initialize_serial()

        self.plot = DynamicPlot()
        
    def initialize_serial(self) -> serial.Serial:
        """
        Initialize and return the serial connection.

        Returns:
            Serial: The initialized serial connection.
        """
        ser = serial.Serial(self.RealTimeconfig['serial_port'], self.RealTimeconfig['serial_baudrate'])
        for _ in range(100):
            ser.readline()
        return ser

    def load_test_data(self) -> None:
        """Load test data from files."""
        self.data = pd.read_csv(self.RealTimeconfig['data_path'])
        self.testInputs = self.data[self.RealTimeconfig['input_featurs']].to_numpy()
        self.testInputsLabes = self.data[self.RealTimeconfig['input_labels']].to_numpy()
        self.testInputs = self.feature_scaler.transform(self.testInputs)
        self.testInputs_seq = create_sliding_sequences(self.testInputs, self.config.sequence_length)
        self.testInputs_labels_seq = create_sliding_sequences(self.testInputsLabes, self.config.sequence_length)
        self.testIndex = 0

    def run(self) -> None:
        """Main loop for real-time data handling."""
        while True:
            start_time = time.time()
            sequence = self.read_seq()
            if sequence is not None and len(sequence) > 0:
                prediction = self.predict(sequence)
                if prediction is not None and len(prediction) > 0:
                    prediction = self.apply_savgol_filter(prediction)
                    ground_truth = self.get_ground_truth()
                    self.plot.update_plot(prediction, ground_truth)
                    end_time = time.time()
                    print("Time taken: ", end_time - start_time)
                else:
                    logger.warning("Prediction was None")
            else:
                logger.warning("Sequence was None")

if __name__ == "__main__":
    config_path = 'RealTimeConfig.yaml'
    realtime = RealTimeSystem(config_path)
    realtime.natnet_reader.natnet.run()
    realtime.run()
    realtime.natnet_reader.natnet.stop()