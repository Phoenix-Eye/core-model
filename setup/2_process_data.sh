#!/bin/bash

echo "🔄 Starting data processing..."

# Activate virtual environment
source venv/bin/activate

echo "⚙️ Processing raw data..."
python << EOF
from src.data.processor import WildfireDataProcessor
from src.config import config
import numpy as np
from pathlib import Path

try:
    # Load raw data
    raw_dir = Path('data/raw')
    processor = WildfireDataProcessor(config)
    
    # Process data if it exists
    if raw_dir.exists() and any(raw_dir.iterdir()):
        processed_data = processor.prepare_data(raw_dir)
    else:
        # Generate sample data if no raw data exists
        processed_data = {
            'spatial': np.random.random((100, 64, 64, 5)),
            'temporal': np.random.random((100, 24, 10)),
            'labels': np.random.randint(0, 2, (100, 64, 64))
        }
    
    # Save processed data
    processed_dir = Path('data/processed')
    processed_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        processed_dir / 'processed_data.npz',
        **processed_data
    )
except Exception as e:
    print(f"Error processing data: {str(e)}")
    exit(1)
EOF

if [ $? -ne 0 ]; then
    echo "❌ Data processing failed. Check logs for details."
    exit 1
fi

echo "✅ Data processing complete!"