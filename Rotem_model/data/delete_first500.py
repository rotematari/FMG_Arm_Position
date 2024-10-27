import os
import pandas as pd

# Path to the folder containing CSV files
folder_path = 'Rotem_model/data/new'

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        # Construct full file path
        file_path = os.path.join(folder_path, filename)
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        # Delete the first 500 rows
        df = df.iloc[500:]
        
        # Save the modified DataFrame back to the CSV file
        df.to_csv(file_path, index=False)
        
        print(f"Processed {filename}")

print("All files processed.")
