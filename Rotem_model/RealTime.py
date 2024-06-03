import torch
from data.data_processing import DataProcessor
from sklearn.preprocessing import StandardScaler
import numpy as np 
from models.models import CNNLSTMModel,Conv2DLSTMAttentionModel,TransformerModel
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

        self.first = True
        self.testFromFlie = False

        if self.testFromFlie:
            self.testInputs = np.load('inputs.npy')
            self.testInputsLabes = np.load('inputs_labels.npy',allow_pickle=True)
            self.testInputs_seq = self.create_sliding_sequences(self.testInputs,self.config.sequence_length)
            self.testInputs_labels_seq = self.create_sliding_sequences(self.testInputsLabes,self.config.sequence_length)
            self.testIndex = 0
        else:
            self.ser = self.initialize_serial()
            if False:
                calibration_length = 1000
                print("start clibration \n ---------------- \n")

                self.calibrate_system(calibration_length=calibration_length)
                
                print("end clibration \n ---------------- \n")
        self.session_bias =0
        self.last_Data_Point = [0 for i in range(32)]

        self.plot = DynamicPlot()

    def initialize_model(self):
        # Initialize and return the model
        return TransformerModel(self.config).to(self.device)
    
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
            # start =time.time()
            line = self.ser.readline().decode("utf-8").rstrip(',\r\n') # Read a line from the serial port
            line = line.split(',')
            if len(line) == 32 :
                DataPoint = [int(num) for num in line]
                self.last_Data_Point = DataPoint
            else:
                DataPoint = self.last_Data_Point
                print("bad reading")

        except Exception as e:
            print(f'eror in reading sequence : {e}')
            DataPoint = self.last_Data_Point
        # print(f"time to read line {time.time()-start}")
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
    def create_sliding_sequences(self,input_array, sequence_length):
        sample_size, features = input_array.shape
        new_sample_size = sample_size - sequence_length + 1

        sequences = []

        for i in range(new_sample_size):
            sequence = input_array[i:i+sequence_length]
            sequences.append(sequence)

        return np.array(sequences)
    def read_seq(self):
        # start =time.time()
        window_size = self.config.window_size
        sequence_length = self.config.sequence_length +  window_size -1
        
        sequence = []
        if not self.testFromFlie:
            if self.first:
                for i in range(sequence_length):
                    sequence.append(self.readline())
                self.last_sequence = sequence
                self.first = False
            else:
                self.last_sequence = self.last_sequence[1:]
                self.last_sequence.append(self.readline())
                sequence = self.last_sequence
            try:
                #calibrate
                # sequence = sequence- self.session_bias
                # moving averege 
                sequence = convolve2d(np.array(sequence).T,np.ones((1,window_size))/window_size,'valid').T
                #scale
                sequence = self.feature_scaler.transform(sequence)
                # print(sequence.shape)
            except Exception as e:  
                print(f'eror in sequence process : {e}') 
                sequence = np.zeros((200,32))
        else:
            sequence = self.testInputs_seq[self.testIndex]
            self.testIndex += 1
            # time.sleep(0.01)
        # print(f"time to read sequence {time.time()-start}")
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


    def init_plot(self):
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
        self.ax.view_init(elev=30, azim=45)
    def visualize_prediction(self, prediction,true):
        # prediction => [MS(xyz),ME(xyz),MW(xyz)]

        # Visualization logic
        try : 
            shoulder_pred = prediction[0,:3]
            elbow_pred = prediction[0,3:6]
            wrist_pred = prediction[0,6:9]

            shoulder_true = true['shoulder'][0]
            elbow_true = true['elbow'][0]
            wrist_true = true['wrist'][0]

            # Update line data for predictions and true data
            self.line_pred.set_data([shoulder_pred[0], elbow_pred[0], wrist_pred[0]], [shoulder_pred[1], elbow_pred[1], wrist_pred[1]])
            self.line_pred.set_3d_properties([shoulder_pred[2], elbow_pred[2], wrist_pred[2]])

            self.line_true.set_data([shoulder_true[0], elbow_true[0], wrist_true[0]], [shoulder_true[1], elbow_true[1], wrist_true[1]])
            self.line_true.set_3d_properties([shoulder_true[2], elbow_true[2], wrist_true[2]])

            
            
            plt.pause(0.000000001)

        except Exception as e :

            print(f'eror in visualization : {e}')
    
    def run(self):

        while True:  # Main loop for real-time data handling
            # start = time.time()
            # start_read_time = time.time()
            sequence = self.read_seq()
            # end_read_time = time.time()
            # print(f"time to read sequence {end_read_time-start_read_time} sec")
            # start_pred_time = time.time()
            prediction = self.predict(sequence)
            # end_pred_time = time.time()
            # print(f"time to predict {end_pred_time-start_pred_time} sec")
            start_visualize_time = time.time()
            try:
                if self.testFromFlie:
                    ground_truth = {
                                    'chest':[(0,0,0)],
                                    'shoulder':[self.testInputs_labels_seq[self.testIndex,-1,0:3]],
                                    'elbow':[self.testInputs_labels_seq[self.testIndex,-1,3:6]],
                                    'wrist':[self.testInputs_labels_seq[self.testIndex,-1,6:9]],
                                    'table_base':[(0,0,0)],
                                    }
                else:
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
            # self.visualize_prediction(prediction,ground_truth)
            self.plot.update_plot(prediction,ground_truth)
            # end_visualize_time = time.time()
            # print(f"time to visualize {end_visualize_time-start_visualize_time} sec")
            # end = time.time()
            # print(f"all {end-start} sec")
            # self.visualize_prediction(ground_truth)

class DynamicPlot:
    def __init__(self):
        # Initialize plot [-0.14760526, -0.3198805 ,  0.57936525, -0.11891279, -0.03982576,
        #  0.42112935, -0.06532572,  0.20495313,  0.37788713]
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.line_pred, = self.ax.plot([], [], [], 'r-', label='Prediction')
        self.line_true, = self.ax.plot([], [], [], 'b-', label='True')
        plt.legend()

        self.ax.set_xlabel('X axis')
        self.ax.set_ylabel('Y axis')
        self.ax.set_zlabel('Z axis')
        self.ax.set_xlim([-40,55])
        self.ax.set_ylim([-50,50])
        self.ax.set_zlim([-50,60])

        # self.ax.view_init(elev=30, azim=45)
        
        plt.show(block=False)
        plt.pause(0.0001)
        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self.ax.draw_artist(self.line_pred)
        self.ax.draw_artist(self.line_true)
        self.fig.canvas.blit(self.fig.bbox)

    def update_plot(self, new_data_pred, new_data_true):

        shoulder_pred = new_data_pred[0,:3]
        elbow_pred = new_data_pred[0,3:6]
        wrist_pred = new_data_pred[0,6:9]

        shoulder_true = new_data_true['shoulder'][0]
        elbow_true = new_data_true['elbow'][0]
        wrist_true = new_data_true['wrist'][0]

        # Update line data for predictions and true data
        self.line_pred.set_data([shoulder_pred[0], elbow_pred[0], wrist_pred[0]], [shoulder_pred[1], elbow_pred[1], wrist_pred[1]])
        self.line_pred.set_3d_properties([shoulder_pred[2], elbow_pred[2], wrist_pred[2]])

        self.line_true.set_data([shoulder_true[0]*100, elbow_true[0]*100, wrist_true[0]*100], [shoulder_true[1]*100, elbow_true[1]*100, wrist_true[1]*100])
        self.line_true.set_3d_properties([shoulder_true[2]*100, elbow_true[2]*100, wrist_true[2]*100])
        # # Update data
        # self.line_pred.set_data(new_data_pred[0,:3], new_data_pred[0,3:6])
        # self.line_pred.set_3d_properties(new_data_pred[])
        
        # self.line_true.set_data(new_data_true[0], new_data_true[1])
        # self.line_true.set_3d_properties(new_data_true[2])
            # reset the background back in the canvas state, screen unchanged
        self.fig.canvas.restore_region(self.bg)
        # Redraw the lines using blit
        self.ax.draw_artist(self.line_pred)
        self.ax.draw_artist(self.line_true)
        self.fig.canvas.blit(self.ax.bbox)
        self.fig.canvas.flush_events()








if __name__ == "__main__":

    model_check_point = r'models/saved_models/TransformerModel_epoch_0_date_21_04_09_58.pt'
    realtime = RealTimeSystem(model_check_point)
    realtime.natnet_reader.natnet.run()
    realtime.run()
    realtime.natnet_reader.natnet.stop()