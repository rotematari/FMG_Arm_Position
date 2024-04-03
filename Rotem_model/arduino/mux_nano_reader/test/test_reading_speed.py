import serial
import time

# Replace with your serial port details
serial_port = '/dev/ttyACM0'
baud_rate = 115200

# Initialize serial connection
ser = serial.Serial(serial_port, baud_rate)

print("Starting to read 1000 lines from the serial port...")

total_time = 0
num_lines = 1000
ser.readline()
for i in range(num_lines):
    start_time = time.perf_counter()  # Start the timer
    
    line = ser.readline().decode("utf-8").rstrip(',\r\n')# Read a line from the serial port
    numbers = [int(num) for num in line.split(',')]
    end_time = time.perf_counter()  # Stop the timer
    read_time = (end_time - start_time)  # Time in seconds
    total_time += read_time  # Accumulate the total time

# Calculate the average time in microseconds
average_time = (total_time / num_lines) 

frequency_hz = 1 / average_time

print(f"Average reading speed: {frequency_hz} Hz")
print(numbers)
print(len(numbers))

