# scripts/train.py
import os
import sys
import argparse
import logging
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import tensorflow as tf

# Add project root to path
root_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(root_dir))

from src.config import config
from src.data.collector import WildfireDataCollector
from src.data.processor import WildfireDataProcessor
from src.models.wildfire_model import WildfirePredictionModel

def setup_logging(log_path: str) -> logging.Logger:
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train wildfire prediction model')
    parser.add_argument(
        '--data-path',
        type=str,
        help='Path to preprocessed data'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to save trained model'
    )
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Training batch size'
    )
    parser.add_argument(
        '--num-ensemble',
        type=int,
        default=3,
        help='Number of ensemble models'
    )
    return parser.parse_args()

def prepare_training_data(data_path: str = None):
    """Prepare data for training"""
    if data_path and Path(data_path).exists():
        # Load preprocessed data
        data = np.load(data_path, allow_pickle=True).item()
        return data
        
    # If no data path provided or file doesn't exist, collect new data
    collector = WildfireDataCollector(config['data'].region_bounds)
    processor = WildfireDataProcessor(config)
    
    # Collect historical data
    raw_data = collector.collect_all_data(
        start_date='2020-01-01',  # Adjust date range as needed
        end_date='2023-12-31'
    )
    
    # Process data
    processed_data = processor.prepare_data(raw_data)
    
    return processed_data

def train_model(
    model,
    data,
    num_epochs: int,
    batch_size: int,
    model_path: str,
    logger: logging.Logger
):
    """Train the model and save results"""
    # Split data into train/validation
    train_idx = int(len(data['spatial']) * 0.8)
    
    train_data = {
        'spatial': data['spatial'][:train_idx],
        'temporal': data['temporal'][:train_idx],
        'labels': data['labels'][:train_idx]
    }
    
    val_data = {
        'spatial': data['spatial'][train_idx:],
        'temporal': data['temporal'][train_idx:],
        'labels': data['labels'][train_idx:]
    }
    
    # Train model
    histories = model.fit(
        x=[train_data['spatial'], train_data['temporal']],
        y=train_data['labels'],
        validation_data=(
            [val_data['spatial'], val_data['temporal']],
            val_data['labels']
        ),
        batch_size=batch_size,
        epochs=num_epochs
    )
    
    # Save training history
    history_path = Path(model_path) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(histories, f)
    
    # Save models
    model.save_models(model_path)
    
    return histories

def evaluate_model(model, data, logger: logging.Logger):
    """Evaluate model performance"""
    # Make predictions
    predictions, uncertainties = model.predict(
        x=[data['spatial'], data['temporal']],
        return_uncertainty=True
    )
    
    # Calculate metrics
    binary_predictions = (predictions > 0.5).astype(int)
    accuracy = np.mean(binary_predictions == data['labels'])
    
    # Calculate confidence intervals
    confidence = 1 - uncertainties
    mean_confidence = np.mean(confidence)
    
    logger.info(f"Model Accuracy: {accuracy:.4f}")
    logger.info(f"Mean Confidence: {mean_confidence:.4f}")
    
    return {
        'accuracy': accuracy,
        'mean_confidence': mean_confidence
    }

def main():
    args = parse_args()
    
    # Setup logging
    log_path = Path('logs') / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logging(log_path)
    
    try:
        # Prepare data
        logger.info("Preparing training data...")
        data = prepare_training_data(args.data_path)
        
        # Initialize model
        logger.info("Initializing model...")
        model = WildfirePredictionModel(
            config=config['model'],
            num_ensemble=args.num_ensemble
        )
        model.build_ensemble()
        
        # Train model
        logger.info("Starting training...")
        histories = train_model(
            model=model,
            data=data,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            model_path=args.model_path,
            logger=logger
        )
        
        # Evaluate model
        logger.info("Evaluating model...")
        metrics = evaluate_model(model, data, logger)
        
        # Save evaluation metrics
        metrics_path = Path(args.model_path) / 'evaluation_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)
            
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()