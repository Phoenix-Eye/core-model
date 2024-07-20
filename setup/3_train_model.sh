#!/bin/bash

echo "🧠 Starting model training..."

# Activate virtual environment
source venv/bin/activate

# Create models directory if it doesn't exist
mkdir -p models/saved

echo "🚀 Training model..."
python scripts/train.py \
    --data-path "data/processed/processed_data.npz" \
    --model-path "models/saved" \
    --num-epochs 100 \
    --batch-size 32 \
    --num-ensemble 3

if [ $? -ne 0 ]; then
    echo "❌ Training failed. Check logs for details."
    exit 1
fi

echo "✅ Model training complete!"