#!/bin/bash

echo "📥 Starting data collection process..."

# Activate virtual environment
source venv/bin/activate

# Create data directories if they don't exist
mkdir -p data/raw
mkdir -p data/processed
mkdir -p logs

# Check if Earth Engine credentials exist
if [ ! -f "$HOME/.config/earthengine/credentials" ]; then
    echo "🔑 Earth Engine credentials not found. Setting up authentication..."
    
    # Try to authenticate with Earth Engine
    python << EOF
import ee
try:
    ee.Authenticate()
    print("✅ Authentication successful!")
except Exception as e:
    print(f"⚠️  Authentication failed: {str(e)}")
    print("Continuing with sample data mode...")
EOF
fi

echo "🌍 Collecting data..."
python << EOF
from src.data.collector import WildfireDataCollector
from src.config import config
import os

try:
    collector = WildfireDataCollector(config['data'].region_bounds)
    data = collector.collect_all_data(
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    print("✅ Data collection successful")
except Exception as e:
    print(f"⚠️  Error during data collection: {str(e)}")
    if not os.path.exists('data/processed/processed_data.npz'):
        print("Generating sample data...")
        collector = WildfireDataCollector({'lat_min': 0, 'lat_max': 1, 'lon_min': 0, 'lon_max': 1})
        data = collector._generate_sample_data()
EOF

# Check if data was collected
if [ ! -f "data/processed/processed_data.npz" ]; then
    echo "⚠️  No data file found. Generating sample data..."
    python << EOF
import numpy as np
from pathlib import Path

# Generate sample data
sample_data = {
    'spatial': np.random.random((100, 64, 64, 5)),
    'temporal': np.random.random((100, 24, 10)),
    'labels': np.random.randint(0, 2, (100, 64, 64))
}

# Save sample data
processed_dir = Path('data/processed')
processed_dir.mkdir(parents=True, exist_ok=True)
np.savez(processed_dir / 'processed_data.npz', **sample_data)
print("✅ Sample data generated and saved")
EOF
fi

echo "✅ Data collection step complete!"