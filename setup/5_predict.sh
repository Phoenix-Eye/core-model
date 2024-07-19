#!/bin/bash

echo "🔮 Starting prediction service..."

# Activate virtual environment
source venv/bin/activate

echo "🎯 Making predictions..."
python scripts/predict.py \
    --model-path "models/saved" \
    --region "nogales" \
    --output-path "predictions"

echo "✅ Predictions complete!"