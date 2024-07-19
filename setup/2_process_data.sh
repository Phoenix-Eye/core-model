#!/bin/bash

echo "🔄 Starting data processing..."

# Activate virtual environment
source venv/bin/activate

echo "⚙️ Processing raw data..."
python src/data/processor.py \
    --input-dir "data/raw" \
    --output-dir "data/processed" \
    --grid-size 64

echo "✅ Data processing complete!"