import torch
import yaml
import logging
from sklearn.preprocessing import StandardScaler
import numpy as np 
import pandas as pd
from models.models import TransformerModel
from scipy.signal import convolve2d
import serial
import matplotlib.pyplot as plt
import os
from Real_Time.NatnetReader import NatNetReader
from scipy.signal import savgol_filter
import time

# Change the current working directory to the directory of the script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealTimeSystem:
    """
    A class to handle real-time data processing, prediction, and visualization using a neural network model.
    """

    def __init__(self, config_path):
        """
        Initialize the RealTimeSystem with a configuration file.

        Args:
            config_path (str): Path to the configuration YAML file.
        """
        self.load_config(config_path)
        
        self.natnet_reader = NatNetReader()
        self.device = self.initialize_device()

        self.load_checkpoint(self.RealTimeconfig['checkpoint_path'])
        self.config = self.checkpoint['config']
        self.model = self.initialize_model()
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.feature_scaler, self.label_scaler = self.initialize_scaler()

        self.first = True
        
        if self.RealTimeconfig['testFromFile']:
            self.load_test_data()
        else:
            self.ser = self.initialize_serial()
            # self.calibrate_system_if_needed()

        self.plot = DynamicPlot()
        
    def load_config(self, config_path):
        """
        Load configuration from a YAML file.

        Args:
            config_path (str): Path to the configuration YAML file.
        """
        with open(config_path, 'r') as file:
            self.RealTimeconfig = yaml.safe_load(file)

    def initialize_device(self):
        """Initialize the computation device (CPU or GPU)."""
        if self.RealTimeconfig["device"]== "cuda" and torch.cuda.is_available():
            logger.info("CUDA is available. Using GPU: %s", torch.cuda.get_device_name(0))
            return torch.device("cuda:0")
        else:
            logger.info("CUDA is not available or CPU is selected. Using CPU.")
            return torch.device("cpu")

    def initialize_model(self):
        """
        Initialize and return the model.

        Returns:
            model: The initialized model.
        """
        return TransformerModel(self.config).to(self.device)
    
    def initialize_serial(self):
        """
        Initialize and return the serial connection.

        Returns:
            Serial: The initialized serial connection.
        """
        ser = serial.Serial(self.RealTimeconfig['serial_port'], self.RealTimeconfig['serial_baudrate'])
        for _ in range(100):
            ser.readline()
        return ser 
    
    def initialize_scaler(self):
        """
        Initialize and return the feature and label scalers.

        Returns:
            tuple: A tuple containing the feature and label scalers.
        """
        label_scaler = StandardScaler()
        feature_scaler = StandardScaler()
        std_feature_scaler_state = self.checkpoint['std_feature_scaler_state']
        std_label_scaler_state = self.checkpoint['std_label_scaler_state']
        
        label_scaler.mean_ = np.array(std_label_scaler_state['mean'])
        label_scaler.var_ = np.array(std_label_scaler_state['var'])
        label_scaler.scale_ = np.array(std_label_scaler_state['scale'])
        label_scaler.n_samples_seen_ = std_label_scaler_state['n_samples_seen']
        
        feature_scaler.mean_ = np.array(std_feature_scaler_state['mean'])
        feature_scaler.var_ = np.array(std_feature_scaler_state['var'])
        feature_scaler.scale_ = np.array(std_feature_scaler_state['scale'])
        feature_scaler.n_samples_seen_ = std_feature_scaler_state['n_samples_seen']
        
        return feature_scaler, label_scaler

    def load_checkpoint(self, checkpoint_path):
        """
        Load the model checkpoint.

        Args:
            checkpoint_path (str): Path to the model checkpoint file.
        """
        self.checkpoint = torch.load(checkpoint_path)

    def load_test_data(self):
        """Load test data from files."""
        self.data = pd.read_csv(self.RealTimeconfig['data_path'])
        self.testInputs = self.data[self.RealTimeconfig['input_featurs']].to_numpy()
        self.testInputsLabes = self.data[self.RealTimeconfig['input_labels']].to_numpy()
        self.testInputs = self.feature_scaler.transform(self.testInputs)
        self.testInputs_seq = self.create_sliding_sequences(self.testInputs, self.config.sequence_length)
        self.testInputs_labels_seq = self.create_sliding_sequences(self.testInputsLabes, self.config.sequence_length)
        self.testIndex = 0

    def calibrate_system_if_needed(self):
        """Calibrate the system if needed."""
        calibration_length = self.RealTimeconfig['calibration_length']
        logger.info("Start calibration\n----------------\n")
        self.calibrate_system(calibration_length=calibration_length)
        logger.info("End calibration\n----------------\n")

    def calibrate_system(self, calibration_length):
        """
        Read calibration data and set up bias and scale parameters.

        Args:
            calibration_length (int): Number of data points for calibration.
        """
        Caldata = []

        for i in range(calibration_length):
            Caldata.append(self.readline())
            if i % 100 == 0:
                logger.info('%d points collected', i)

        Caldata = np.array(Caldata, dtype=np.float32)
        self.session_bias = Caldata.mean(axis=0)

    def create_sliding_sequences(self, input_array: np.ndarray, sequence_length: int):
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
    
    def readline(self):
        """
        Read a line of data from the serial connection.

        Returns:
            list: A list of data points.
        """
        try:
            self.ser.reset_input_buffer()
            
            line = self.ser.readline().decode("utf-8").rstrip(',\r\n').split(',')
            line = self.ser.readline().decode("utf-8").rstrip(',\r\n').split(',')
            
            if len(line) == 32:
                DataPoint = [int(num) for num in line]
                # print(DataPoint[-1])
                # self.last_Data_Point = DataPoint
                return DataPoint[:32]
            else:
                logger.warning("Bad reading: Expected 32 data points, got %d", len(line))
                return None
        except Exception as e:
            logger.error('Error reading line: %s', e)
            return None

    def read_seq(self):
        """
        Read a sequence of data points.

        Returns:
            numpy.ndarray: Processed sequence of data points.
        """
        window_size = self.config.window_size
        sequence_length = self.config.sequence_length + window_size - 1
        
        if not self.RealTimeconfig['testFromFile']:
            sequence = self.initialize_sequence(sequence_length)
            sequence = self.process_sequence(sequence, window_size)
        else:
            self.testInputs
            sequence = self.testInputs_seq[self.testIndex]
            self.testIndex += 1

        return sequence

    def get_good_reading(self):
        """
        Keep trying to get a good reading.

        Returns:
            list: A valid data point.
        """
        while True:
            line = self.readline()
            if line:
                return line
    
    def initialize_sequence(self, sequence_length):
        """
        Initialize the sequence for real-time data.

        Args:
            sequence_length (int): The length of the sequence.

        Returns:
            list: Initialized sequence.
        """
        sequence = []
        if self.first:
            while len(sequence) < sequence_length:
                sequence.append(self.get_good_reading())
            self.last_sequence = sequence
            self.first = False
        else:
            new_reading = self.get_good_reading()
            self.last_sequence = self.last_sequence[1:]
            self.last_sequence.append(new_reading)
            
        return self.last_sequence
    def numpy_ewma(self,data, window):
        alpha = 2 / (window + 1)
        alpha_rev = 1 - alpha
        n = data.shape[0]

        pows = alpha_rev**(np.arange(n+1))
        
        scale_arr = 1/pows[:-1]
        offset = data*pows[1:]
        pw0 = alpha*alpha_rev**(n-1)
        
        mult = data*pw0*scale_arr
        cumsums = mult.cumsum()
        out = offset + cumsums*scale_arr[::-1]
        
        return out
    def process_sequence(self, sequence, window_size):
        """
        Process the sequence by applying calibration, moving average, and scaling.

        Args:
            sequence (list): The sequence to process.
            window_size (int): The window size for moving average.

        Returns:
            numpy.ndarray: Processed sequence.
        """
        try:
            sequence = self.feature_scaler.transform(sequence)
        except Exception as e:
            logger.error('Error in process sequence : %s', e)
            sequence = None
        return sequence

    def predict(self, sequence):
        """
        Predict using the model.

        Args:
            sequence (numpy.ndarray): Input sequence.

        Returns:
            numpy.ndarray: Model prediction.
        """
        self.model.eval()
        with torch.no_grad():

            sequence = torch.tensor(sequence, dtype=torch.float32, device=self.device).unsqueeze(0)
            try:
                prediction = self.model(sequence)
                prediction = prediction[:, -1, :]
                prediction = self.label_scaler.inverse_transform(prediction.detach().cpu().numpy())
            except Exception as e:
                logger.error('Error in prediction: %s', e)
                prediction = None

        return prediction
    def numpy_ewma_vectorized_v2(self,data, window):
        alpha = 2 / (window + 1.0)
        alpha_rev = 1 - alpha
        n = data.shape[0]

        pows = alpha_rev**(np.arange(n + 1))

        scale_arr = 1 / pows[:-1]
        offset = data[0] * pows[1:]
        pw0 = alpha * alpha_rev**(n - 1)

        mult = data * pw0 * scale_arr
        cumsums = mult.cumsum()
        out = offset + cumsums * scale_arr[::-1]
        return out

    def apply_numpy_ewma(self,data, window):
        batch, sequence, features = data.shape
        result = np.zeros_like(data)
        for b in range(batch):
            for f in range(features):
                result[b, :, f] = self.numpy_ewma_vectorized_v2(data[b, :, f], window)
        return result
    
    def apply_savgol_filter(self,prediction: np.ndarray)-> np.ndarray:
        
        if hasattr(self, 'pastPredictions'):
            if self.pastPredictions.shape[0] < self.RealTimeconfig["maxpastsize"]:
                self.pastPredictions = np.vstack((self.pastPredictions,prediction))
            else:
                self.pastPredictions = np.vstack((self.pastPredictions[1:],prediction))
        else:
            self.pastPredictions = prediction
        
        if self.pastPredictions.shape[0] > 5:
            #calc slagov filter loc Vel
            if self.pastPredictions.shape[0]<100:
                window_size = self.pastPredictions.shape[0]
            else:
                window_size = 100
            
            poly_order = 5
            # pastPredictions = savgol_filter(self.pastPredictions,deriv=0, window_length=window_size, polyorder=poly_order,axis=0)
            firstOrder_derivative_predsToPlot = savgol_filter(self.pastPredictions,deriv=1,delta=30.0, window_length=window_size, polyorder=poly_order,axis=0)
            # # calc new loc 
            # prediction = pastPredictions[-1] 
            prediction += firstOrder_derivative_predsToPlot[-1]*100

        return prediction
    def run(self):
        """Main loop for real-time data handling."""
        
        while True:
            starTime = time.time()
            sequence = self.read_seq()
            if sequence is not None and len(sequence) > 0:
                prediction = self.predict(sequence)
                if prediction is not None and len(prediction) > 0:
                    prediction = self.apply_savgol_filter(prediction)
                    ground_truth = self.get_ground_truth()
                    self.plot.update_plot(prediction, ground_truth)
                    endTime = time.time()
                    print("Time taken: ", endTime - starTime)
                else:
                    logger.warning("prediction was None")
            else:
                logger.warning("sequence was None")

    def get_ground_truth(self):
        """
        Get ground truth data.

        Returns:
            dict: Ground truth data.
        """
        if self.RealTimeconfig['testFromFile']:
            ground_truth = {
                'chest': [(0, 0, 0)],
                'shoulder': [self.testInputs_labels_seq[self.testIndex, -1, 0:3]],
                'elbow': [self.testInputs_labels_seq[self.testIndex, -1, 3:6]],
                'wrist': [self.testInputs_labels_seq[self.testIndex, -1, 6:9]],
                'table_base': [(0, 0, 0)],
            }
        else:
            try:
                ground_truth = self.natnet_reader.read_sample()
                if  len(ground_truth['shoulder']) == 0:
                    ground_truth = {
                    'chest': [(0, 0, 0)],
                    'shoulder': [(0, 0, 0)],
                    'elbow': [(0, 0, 0)],
                    'wrist': [(0, 0, 0)],
                    'table_base': [(0, 0, 0)],
                }

            except Exception as e:
                logger.error('Error in NatNetReader: %s', e)
                ground_truth = {
                    'chest': [(0, 0, 0)],
                    'shoulder': [(0, 0, 0)],
                    'elbow': [(0, 0, 0)],
                    'wrist': [(0, 0, 0)],
                    'table_base': [(0, 0, 0)],
                }
        return ground_truth


class DynamicPlot:
    def __init__(self):
        # Initialize plot
        self.fig = plt.figure(figsize=[50,50])
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.line_pred, = self.ax.plot([], [], [], 'r-', label='Prediction')
        self.line_true, = self.ax.plot([], [], [], 'b-', label='True')
        plt.legend()

        self.ax.set_xlabel('X axis')
        self.ax.set_ylabel('Y axis')
        self.ax.set_zlabel('Z axis')
        self.ax.set_xlim([-40, 60])
        self.ax.set_ylim([-50, 50])
        self.ax.set_zlim([-50, 60])
        
        # Set the view angle
        self.ax.view_init(elev=30, azim=10)# Adjust the angles as needed

        plt.show(block=False)
        plt.pause(0.00001)

        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self.ax.draw_artist(self.line_pred)
        self.ax.draw_artist(self.line_true)
        self.fig.canvas.blit(self.fig.bbox)

    def update_plot(self, new_data_pred: np.ndarray, new_data_true: dict):
        new_data_pred = new_data_pred.reshape((-1))
        # shoulder_pred = new_data_pred[0, :3]
        # elbow_pred = new_data_pred[0, 3:6]
        # wrist_pred = new_data_pred[0, 6:9]
        shoulder_pred = new_data_pred[:3]
        elbow_pred = new_data_pred[3:6]
        wrist_pred = new_data_pred[6:9]

        # shoulder_true = new_data_true['shoulder'][0]*100
        # elbow_true = new_data_true['elbow'][0]*100
        # wrist_true = new_data_true['wrist'][0]*100

        shoulder_true = tuple(x * 100 for x in new_data_true['shoulder'][0])
        elbow_true = tuple(x * 100 for x in new_data_true['elbow'][0])
        wrist_true = tuple(x * 100 for x in new_data_true['wrist'][0])

        self.line_pred.set_data([shoulder_pred[0], elbow_pred[0], wrist_pred[0]], [shoulder_pred[1], elbow_pred[1], wrist_pred[1]])
        self.line_pred.set_3d_properties([shoulder_pred[2], elbow_pred[2], wrist_pred[2]])

        self.line_true.set_data([shoulder_true[0], elbow_true[0], wrist_true[0]], [shoulder_true[1], elbow_true[1], wrist_true[1]])
        self.line_true.set_3d_properties([shoulder_true[2], elbow_true[2], wrist_true[2]])

        self.fig.canvas.restore_region(self.bg)
        self.ax.draw_artist(self.line_pred)
        self.ax.draw_artist(self.line_true)
        self.fig.canvas.blit(self.ax.bbox)
        self.fig.canvas.flush_events()


if __name__ == "__main__":
    config_path = 'RealTimeConfig.yaml'
    realtime = RealTimeSystem(config_path)
    realtime.natnet_reader.natnet.run()
    realtime.run()
    realtime.natnet_reader.natnet.stop()
