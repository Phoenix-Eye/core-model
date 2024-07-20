#!/bin/bash

echo "📥 Starting data collection process..."

# Activate virtual environment
source venv/bin/activate

# Create data directories if they don't exist
mkdir -p data/raw
mkdir -p data/processed

echo "🌍 Collecting satellite data..."
python << EOF
from src.data.collector import WildfireDataCollector
from src.config import config

collector = WildfireDataCollector(config['data'].region_bounds)
data = collector.collect_all_data(
    start_date='2020-01-01',
    end_date='2023-12-31'
)
EOF

if [ $? -ne 0 ]; then
    echo "❌ Data collection failed. Check logs for details."
    exit 1
fi

echo "✅ Data collection complete!"