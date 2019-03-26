#!/usr/bin/bash

SRC_PATH="/Users/aarontrefler_temp2/Documents/My Documents/Kaggle/kaggle-ncaa-men-19/src/"

export PYTHONPATH="${PYTHONPATH}:${SRC_PATH}"

echo "Running make_dataset..."
python "${SRC_PATH}data/make_dataset.py"

echo "Running build_features..."
python "${SRC_PATH}features/build_features.py"

echo "Running train_model..."
python "${SRC_PATH}models/train_model.py"

echo "Running predict_model..."
python "${SRC_PATH}models/predict_model.py"

echo "Finished running scripts"