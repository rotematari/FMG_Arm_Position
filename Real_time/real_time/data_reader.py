import serial
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DataReader:
    def __init__(self, config):
        self.config = config
        self.serial_port = config['real_time']['serial_port']
        self.serial_baudrate = config['real_time']['serial_baudrate']
        self.expected_data_points = config['real_time']['expected_data_points']
        self.device = serial.Serial(self.serial_port, self.serial_baudrate)
        self.device.flush()
        self.first = True

    def get_good_reading(self):
        while True:
            self.device.reset_input_buffer()
            line = self.device.readline().decode("utf-8").rstrip(',\r\n').split(',')
            line = self.device.readline().decode("utf-8").rstrip(',\r\n').split(',')
            if len(line) == self.expected_data_points:
                return np.array([int(num) for num in line])
            logger.warning("Bad reading: Expected %d data points", self.expected_data_points)
    
    def get_sequence(self):
        """
        Initialize the sequence for real-time data.

        Args:
            sequence_length (int): The length of the sequence.

        Returns:
            list: Initialized sequence.
        """
        sequence_length = self.config['sequence_length']
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
            
        return np.array(self.last_sequence)
    

    
    