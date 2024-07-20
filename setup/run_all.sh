#!/bin/bash

echo "🚀 Running complete Phoenix Eye pipeline..."

# Run each step in sequence
./1_collect_data.sh
./2_process_data.sh
./3_train_model.sh
./4_evaluate_model.sh
./5_predict.sh

echo "✅ Pipeline complete!"