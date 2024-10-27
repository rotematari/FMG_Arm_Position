#!/bin/bash

# Create the directory structure
mkdir -p config models real_time utils

# Create the files
# Configuration file
touch config/real_time_config.yaml

# Transformer model
touch models/transformer_model.py

# Real-time module files
touch real_time/__init__.py
real_time_files=(data_reader.py prediction.py plot.py scaler.py)
for file in "${real_time_files[@]}"; do
  touch real_time/$file
done

# Utility module files
touch utils/__init__.py
util_files=(logger.py serial_manager.py)
for file in "${util_files[@]}"; do
  touch utils/$file
done

# Main entry point
touch main.py

# Print success message
echo "Project structure has been successfully set up."