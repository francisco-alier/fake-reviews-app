#!/bin/bash

# Initial setup
conda activate myenv
cd scr/

# Run EDA
echo "Running eda.py..."
python eda.py

# Run the data cleaning and preprocessing script
echo "Running clean.py..."
python clean.py

# Run the training script
echo "Running train.py..."
python train.py

# Run the evaluation script
echo "Running evaluate.py..."
python evaluate.py

# Run the tests using pytest
#echo "Running tests with pytest..."
#pytest -v


echo "All scripts and tests have been executed."