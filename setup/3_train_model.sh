#!/bin/bash

echo "🧠 Starting model training..."

# Activate virtual environment
source venv/bin/activate

# Create models directory if it doesn't exist
mkdir -p models/saved

echo "🚀 Training model..."
python scripts/train.py \
    --data-path "data/processed" \
    --model-path "models/saved" \
    --num-epochs 100 \
    --batch-size 32 \
    --num-ensemble 3

echo "✅ Model training complete!"