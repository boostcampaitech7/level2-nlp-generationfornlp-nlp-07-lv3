#!/usr/bin/env python3

import os
import subprocess
from datetime import datetime, timedelta

# Get current time (UTC + 9 hours)
current_time = datetime.now() + timedelta(hours=9)
current_time_str = current_time.strftime('%Y%m%d_%H%M%S')

# Root directory (adjust this if necessary)
root_dir = os.getcwd()
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('output', exist_ok=True)

# Ensure root directory exists
if not os.path.exists(root_dir):
    raise FileNotFoundError(f"The root directory {root_dir} does not exist. Please adjust the path accordingly.")

# Change to src directory
src_dir = os.path.join(root_dir, 'src')
if not os.path.exists(src_dir):
    raise FileNotFoundError(f"The source directory {src_dir} does not exist. Please adjust the path accordingly.")
os.chdir(src_dir)

run_name = input('Please Enter Your Run Name : ')

while run_name == '':
    run_name = input('Please Enter Your Run Name : ')

run_name += "_" + current_time_str

# Set up directories
train_dir = os.path.join(root_dir, 'models', f'train_{run_name}')
predict_dir = os.path.join(root_dir, 'output', f'test_{run_name}')
predict_dataset_name = os.path.join(root_dir, 'data', 'test.csv')

# Perform training
subprocess.run([
    "python", "main.py",
    "--output_dir", train_dir,
    "--do_train",
    "--do_eval",
    "--overwrite_output_dir",
    "--run_name", run_name,
    "--quantization", "8",
], check=True)

# Perform prediction (inference)
subprocess.run([
    "python", "main.py",
    "--output_dir", predict_dir,
    "--test_dataset_name", predict_dataset_name,
    "--model_name_or_path", train_dir,
    "--do_predict",
    "--run_name", run_name,
    "--quantization", "8",
], check=True)