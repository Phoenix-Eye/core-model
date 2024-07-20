#!/bin/bash

echo "🧠 Starting model training..."

# Activate virtual environment
source venv/bin/activate

# Create necessary directories
mkdir -p models/saved logs/training

echo "🚀 Training model..."
python3 << EOF
from src.models.wildfire_model import WildfirePredictionModel
from src.config import config
import numpy as np
from pathlib import Path
import json
import sys

try:
    # Load processed data
    data_path = Path('data/processed/processed_data.npz')
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data not found at {data_path}")
        
    data = np.load(data_path)
    processed_data = {
        'spatial': data['spatial'],
        'temporal': data['temporal'],
        'labels': data['labels']
    }
    
    # Split data
    split_idx = int(len(processed_data['labels']) * 0.8)
    train_data = {
        'spatial': processed_data['spatial'][:split_idx],
        'temporal': processed_data['temporal'][:split_idx],
        'labels': processed_data['labels'][:split_idx]
    }
    val_data = {
        'spatial': processed_data['spatial'][split_idx:],
        'temporal': processed_data['temporal'][split_idx:],
        'labels': processed_data['labels'][split_idx:]
    }
    
    # Initialize and train model
    model = WildfirePredictionModel(
        config=config,
        num_ensemble=3,
        uncertainty=True
    )
    model.build_ensemble()
    
    # Train with enhanced metrics
    training_stats = model.advanced_fit(
        x=[train_data['spatial'], train_data['temporal']],
        y=train_data['labels'],
        validation_data=(
            [val_data['spatial'], val_data['temporal']],
            val_data['labels']
        ),
        target_accuracy=0.90,
        max_epochs=1000,
        batch_size=32
    )
    
    # Save model
    model.save_models('models/saved')
    
    # Print training summary
    print("\n📊 Training Summary:")
    print(f"Total Epochs: {training_stats['total_epochs']}")
    print(f"Best Accuracy: {training_stats['best_accuracy']:.4f}")
    print(f"Training Time: {training_stats['training_time']:.2f} seconds")
    print("\nFinal Metrics:")
    for metric, value in training_stats['final_metrics'].items():
        print(f"{metric}: {value:.4f}")
    
    sys.exit(0)
except Exception as e:
    print(f"❌ Training failed: {str(e)}")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo "✅ Model training complete!"
    
    # Display training metrics in run_all.sh output
    if [ -f "logs/training_stats.json" ]; then
        echo "
    ╔════════════════════════════════╗
    ║      Training Metrics          ║
    ╚════════════════════════════════╝
    "
        python3 -c "
import json
with open('logs/training_stats.json', 'r') as f:
    stats = json.load(f)
print(f'Epochs: {stats[\"total_epochs\"]}')
print(f'Best Accuracy: {stats[\"final_metrics\"][\"accuracy\"]:.4f}')
print(f'F1 Score: {stats[\"final_metrics\"][\"f1_score\"]:.4f}')
print(f'ROC AUC: {stats[\"final_metrics\"][\"roc_auc\"]:.4f}')
print(f'Mean Uncertainty: {stats[\"final_metrics\"][\"mean_uncertainty\"]:.4f}')
"
    fi
else
    echo "❌ Training failed. Using sample model..."
    python3 -c "
import numpy as np
from pathlib import Path
save_dir = Path('models/saved')
save_dir.mkdir(parents=True, exist_ok=True)
np.save(save_dir / 'sample_weights.npy', np.random.random((100, 100)))
"
fi