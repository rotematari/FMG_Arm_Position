import os
import pandas as pd

# Specify the directory containing the CSV files
directory = './data/data/full_muvment_clean/'

# Define the new order of columns
new_column_order = [
    'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 
    'S26', 'S27', 'S28', 'S29', 'S30', 'S31', 'S32', 'S33', 'S36', 'S37', 
    'S38', 'S39', 'S40', 'S41', 'S42', 'S43', 'S44', 'S45', 'MCx', 'MCy', 
    'MCz', 'MSx', 'MSy', 'MSz', 'MEx', 'MEy', 'MEz', 'MWx', 'MWy', 'MWz', 
    'sesion_time_stamp'
]

# Iterate over all files in the directory
for filename in os.listdir(directory):
    # Check if the file is a CSV
    if filename.endswith('.csv'):
        # Construct the full file path
        file_path = os.path.join(directory, filename)

        # Read the CSV file
        df = pd.read_csv(file_path)

        # Reorder the columns according to the new_column_order
        df = df[new_column_order]

        # Save the reordered DataFrame back to a new CSV file
        df.to_csv(os.path.join(directory, f'reordered_{filename}'), index=False)

print("All files have been processed and saved with the new column order.")
