#!/bin/bash

echo "🔧 Setting up Phoenix Eye Wildfire Prediction Project..."

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Create necessary directories if they don't exist
mkdir -p data/{raw,processed}
mkdir -p models/saved
mkdir -p logs

echo "✅ Setup complete!"