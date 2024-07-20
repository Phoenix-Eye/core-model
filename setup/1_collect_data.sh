#!/bin/bash

echo "📥 Starting data collection process..."

# Activate virtual environment
source venv/bin/activate

# Set environment variables for API keys
source .env

# Create data directories if they don't exist
mkdir -p data/raw
mkdir -p data/processed

echo "🌍 Collecting satellite data..."
python src/data/collector.py \
    --start-date "2020-01-01" \
    --end-date "2023-12-31" \
    --region "nogales" \
    --output-dir "data/raw"

echo "✅ Data collection complete!"