#!/bin/bash

echo "🚀 Running complete Phoenix Eye pipeline..."

# Run each step in sequence
./setup/1_collect_data.sh
./setup/2_process_data.sh
./setup/3_train_model.sh
./setup/4_evaluate_model.sh
./setup/5_predict.sh

echo "✅ Pipeline complete!"