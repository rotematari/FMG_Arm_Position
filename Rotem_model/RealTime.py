import torch
from data.data_processing import DataProcessor
from sklearn.preprocessing import StandardScaler
import numpy as np 
from models.models import CNNLSTMModel
from scipy.signal import convolve2d
import serial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os 
import time 
from real_time.NatnetReader import NatNetReader

# Change the current working directory to the directory of the script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class RealTimeSystem:
    def __init__(self,checkpoint_path):
        
        # NatNet 
        self.natnet_reader = NatNetReader()




        # Check if CUDA is available
        if torch.cuda.is_available():
            # Set the device to the first CUDA device
            self.device = torch.device("cuda:0")
            print("CUDA is available. Using GPU:", torch.cuda.get_device_name(0))
        else:
            # Fallback to CPU if CUDA is not available
            self.device = torch.device("cpu")
            print("CUDA is not available. Using CPU.")


        self.load_checkpoint(checkpoint_path)
        
        self.config = self.checkpoint['config']
        
        self.model = self.initialize_model()
        self.model.load_state_dict(self.checkpoint['model_state_dict'])

        

        self.feature_scaler,self.label_scaler = self.initialize_scaler()

        # self.data_processor = DataProcessor(self.config)

        # Additional initializations (e.g., visualization tools, data savers) can go here


        # calibration_length = 100
        # print("start clibration \n ---------------- \n")

        # self.calibrate_system(calibration_length=calibration_length)
        # print("end clibration \n ---------------- \n")
        self.first = True
        self.testFromFlie = False
        if self.testFromFlie:
            self.testInputs = np.load('inputs.npy')
            self.testIndex = 0
        else:
            self.ser = self.initialize_serial()
    def initialize_model(self):
        # Initialize and return the model
        return CNNLSTMModel(self.config).to(self.device)
    def initialize_serial(self):
        ser = serial.Serial('/dev/ttyACM0', 115200)
        for i in range(100):
            ser.readline()

        return ser 
    def initialize_scaler(self):
        # Initialize and return the model
        label_scaler = StandardScaler()
        feature_scaler = StandardScaler()
        std_feature_scaler_state = self.checkpoint['std_feature_scaler_state']
        std_label_scaler_state = self.checkpoint['std_label_scaler_state']
        # Restore the state to the new scaler
        ## labels
        label_scaler.mean_ = np.array(std_label_scaler_state['mean'])
        label_scaler.var_ = np.array(std_label_scaler_state['var'])
        label_scaler.scale_ = np.array(std_label_scaler_state['scale'])
        label_scaler.n_samples_seen_ = std_label_scaler_state['n_samples_seen']
        ## feature
        feature_scaler.mean_ = np.array(std_feature_scaler_state['mean'])
        feature_scaler.var_ = np.array(std_feature_scaler_state['var'])
        feature_scaler.scale_ = np.array(std_feature_scaler_state['scale'])
        feature_scaler.n_samples_seen_ = std_feature_scaler_state['n_samples_seen']
        return feature_scaler,label_scaler

    def readline(self):
        # reads a line
        try:
            # print(self.ser.readline().decode("utf-8").rstrip(',\r\n'))
            line = self.ser.readline().decode("utf-8").rstrip(',\r\n') # Read a line from the serial port
            # print(line)
            DataPoint = [int(num) for num in line.split(',')]
            # print(len(DataPoint))
            DataPoint = [DataPoint[i] for i in self.config.sensor_location]

        except Exception as e:
            print(f'eror in reading sequence : {e}')
            DataPoint = [0 for i in range(28)]
        
        # print(data)
        return DataPoint
    
    def calibrate_system(self,calibration_length):
        
        # Read calibration data and set up bias and scale parameters
        Caldata = []
        
        for i in range(calibration_length):
            Caldata.append(self.readline())
            
            if i % 100 == 0:
                print(f'{i} points collected')
        # print(data)
        Caldata = np.array(Caldata, dtype=np.float32)
        
        # Calculate the mean of each column
        self.session_bias = Caldata.mean(axis=0)

    def read_seq(self):

        window_size = self.config.window_size
        sequence_length = self.config.sequence_length +  window_size -1
        
        sequence = []
        if not self.testFromFlie:
            if self.first:
                for i in range(sequence_length):
                    sequence.append(self.readline())
                self.last_sequence = sequence
            else:
                self.last_sequence = self.last_sequence[1:]
                self.last_sequence.append(self.readline())
                sequence = self.last_sequence
            try:
                #calibrate
                # sequence = np.array(sequence)- self.session_bias
                # moving averege 
                sequence = convolve2d(np.array(sequence).T,np.ones((1,window_size))/window_size,'valid').T
                #scale
                sequence = self.feature_scaler.transform(sequence)
            except Exception as e:  
                print(f'eror in sequence process : {e}') 
        else:
            sequence = self.testInputs[self.testIndex]
            self.testIndex += 1
            # time.sleep(0.01)

        # print(sequence)
        return sequence


    def load_checkpoint(self, checkpoint_path):
        # Load the checkpoint and update the model state
        self.checkpoint = torch.load(checkpoint_path)


    def predict(self, sequence):
        self.model.eval()
        with torch.no_grad():
            sequence = torch.tensor(sequence, dtype=torch.float32, device= self.device)
            sequence = sequence.unsqueeze(0)
            try :

                prediction = self.model(sequence)
                prediction = prediction[:,-1,:]
                prediction = self.label_scaler.inverse_transform(prediction.detach().cpu().numpy())
            except Exception as e:
                print(f'eror in prediction : {e}')

        return prediction



    def visualize_prediction(self, prediction,true):
        # prediction => [MS(xyz),ME(xyz),MW(xyz)]
        # prediction = prediction.detach().cpu().numpy()
        # Visualization logic
        try : 
            if self.first:
                # Initialize plot
                self.fig = plt.figure()
                self.ax = self.fig.add_subplot(111, projection='3d')
                self.ax.set_xlabel('X axis')
                self.ax.set_ylabel('Y axis')
                self.ax.set_zlabel('Z axis')
                self.ax.set_xlim([-0.5,0.5])
                self.ax.set_ylim([-0.5,0.5])
                self.ax.set_zlim([-0.5,0.5])
                # Lines to connect points for predictions and true data
                self.line_pred, = self.ax.plot([], [], [], 'ro-', label='Predicted')
                self.line_true, = self.ax.plot([], [], [], 'bo-', label='True')
                self.first = False
            
            shoulder_pred = prediction[0,:3]
            elbow_pred = prediction[0,3:6]
            wrist_pred = prediction[0,6:9]

            shoulder_true = true['shoulder'][0]
            elbow_true = true['elbow'][0]
            wrist_true = true['wrist'][0]


            # line_true, = ax.plot([], [], [], 'bo-', label='True')
            # Update line data for predictions and true data
            self.line_pred.set_data([shoulder_pred[0], elbow_pred[0], wrist_pred[0]], [shoulder_pred[1], elbow_pred[1], wrist_pred[1]])
            self.line_pred.set_3d_properties([shoulder_pred[2], elbow_pred[2], wrist_pred[2]])

            self.line_true.set_data([shoulder_true[0], elbow_true[0], wrist_true[0]], [shoulder_true[1], elbow_true[1], wrist_true[1]])
            self.line_true.set_3d_properties([shoulder_true[2], elbow_true[2], wrist_true[2]])

            self.ax.view_init(elev=30, azim=45)
            
            plt.pause(0.000001)

        except Exception as e :

            print(f'eror in visualization : {e}')
    
    def run(self):

        while True:  # Main loop for real-time data handling
            sequence = self.read_seq()
            prediction = self.predict(sequence)

            try:
                ground_truth = self.natnet_reader.read_sample()
            except Exception as e :
                print(f'eror in natnet{e}')
                ground_truth = {
                'chest':[(0,0,0)],
                'shoulder':[(0,0,0)],
                'elbow':[(0,0,0)],
                'wrist':[(0,0,0)],
                'table_base':[(0,0,0)],
                }
            # print(ground_truth)
            # print(sequence[-1])
            self.visualize_prediction(prediction,ground_truth)
            # self.visualize_prediction(ground_truth)










if __name__ == "__main__":

    model_check_point = r'models/saved_models/CNN_LSTMModel_no_biased.pt'

    realtime = RealTimeSystem(model_check_point)
    realtime.natnet_reader.natnet.run()
    realtime.run()
    realtime.natnet_reader.natnet.stop()