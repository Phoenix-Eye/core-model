import os
import sys
import argparse
import logging
import json
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import numpy as np
import tensorflow as tf
from keras import mixed_precision

# Add project root to path
root_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(root_dir))

from src.config import config
from src.data.collector import WildfireDataCollector
from src.data.processor import WildfireDataProcessor
from src.models.wildfire_model import WildfirePredictionModel

def setup_gpu():
    """Configure GPU and memory settings"""
    # Enable mixed precision
    mixed_precision.set_global_policy('mixed_float16')
    
    # Configure GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logging.info(f"Found {len(gpus)} GPU(s)")
        except RuntimeError as e:
            logging.error(f"GPU configuration error: {e}")
    
    # Enable XLA optimization
    tf.config.optimizer.set_jit(True)

def setup_logging(log_path: str) -> logging.Logger:
    """Configure logging"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
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
        default=200,  # Increased for better convergence
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,  # Reduced for larger grid size
        help='Training batch size'
    )
    parser.add_argument(
        '--num-ensemble',
        type=int,
        default=3,
        help='Number of ensemble models'
    )
    parser.add_argument(
        '--accumulation-steps',
        type=int,
        default=4,
        help='Gradient accumulation steps'
    )
    parser.add_argument(
        '--region',
        type=str,
        choices=list(config['data'].processing_regions().keys()),
        help='Specific region to process'
    )
    return parser.parse_args()

def prepare_training_data(data_path: str = None, region: str = None):
    """Prepare data for training"""
    if data_path and Path(data_path).exists():
        # Load preprocessed data
        data = np.load(data_path, allow_pickle=True).item()
        return data
        
    # If no data path provided or file doesn't exist, collect new data
    collector = WildfireDataCollector(config)
    processor = WildfireDataProcessor(config)
    
    # Collect historical data
    if region:
        bounds = config['data'].processing_regions()[region]
    else:
        bounds = config['data'].default_region_bounds()
    
    raw_data = collector.collect_all_data(
        start_date='2020-01-01',
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
    accumulation_steps: int,
    model_path: str,
    logger: logging.Logger
):
    """Train the model with gradient accumulation and mixed precision"""
    # Setup distribution strategy
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
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
        
        # Calculate effective batch size
        effective_batch_size = batch_size * accumulation_steps
        logger.info(f"Effective batch size: {effective_batch_size}")
        
        # Train with gradient accumulation
        histories = model.advanced_fit(
            x=[train_data['spatial'], train_data['temporal']],
            y=train_data['labels'],
            validation_data=(
                [val_data['spatial'], val_data['temporal']],
                val_data['labels']
            ),
            batch_size=batch_size,
            epochs=num_epochs,
            gradient_accumulation_steps=accumulation_steps
        )
        
        # Save training history
        history_path = Path(model_path) / 'training_history.json'
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, 'w') as f:
            json.dump(histories, f)
        
        # Save models
        model.save_models(model_path)
        
        return histories

def evaluate_model(model, data, logger: logging.Logger, batch_size: int = 16):
    """Evaluate model performance with batched prediction"""
    # Make batched predictions
    total_samples = len(data['spatial'])
    predictions_list = []
    uncertainties_list = []
    
    for i in tqdm(range(0, total_samples, batch_size), desc="Evaluating"):
        batch_slice = slice(i, min(i + batch_size, total_samples))
        batch_pred, batch_unc = model.predict(
            x=[
                data['spatial'][batch_slice],
                data['temporal'][batch_slice]
            ],
            return_uncertainty=True
        )
        predictions_list.append(batch_pred)
        uncertainties_list.append(batch_unc)
    
    # Combine predictions
    predictions = np.concatenate(predictions_list)
    uncertainties = np.concatenate(uncertainties_list)
    
    # Calculate metrics
    binary_predictions = (predictions > 0.5).astype(int)
    accuracy = np.mean(binary_predictions == data['labels'])
    
    # Calculate confidence intervals
    confidence = 1 - uncertainties
    mean_confidence = np.mean(confidence)
    
    # Calculate region-wise metrics
    region_metrics = calculate_regional_metrics(
        predictions, 
        uncertainties, 
        data['labels']
    )
    
    logger.info(f"Overall Accuracy: {accuracy:.4f}")
    logger.info(f"Mean Confidence: {mean_confidence:.4f}")
    
    return {
        'accuracy': float(accuracy),
        'mean_confidence': float(mean_confidence),
        'regional_metrics': region_metrics
    }

def calculate_regional_metrics(predictions, uncertainties, labels):
    """Calculate metrics for different regions of Mexico"""
    regions = config['data'].processing_regions()
    regional_metrics = {}
    
    for region_name, bounds in regions.items():
        # Calculate region-specific metrics
        regional_metrics[region_name] = {
            'accuracy': float(np.mean(
                (predictions > 0.5).astype(int) == labels
            )),
            'uncertainty': float(np.mean(uncertainties)),
        }
    
    return regional_metrics

def main():
    # Parse arguments
    args = parse_args()
    
    # Setup GPU configuration
    setup_gpu()
    
    # Setup logging
    log_path = Path('logs') / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logging(log_path)
    
    try:
        # Log system information
        logger.info(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
        logger.info(f"Mixed precision policy: {mixed_precision.global_policy()}")
        
        # Prepare data
        logger.info("Preparing training data...")
        data = prepare_training_data(args.data_path, args.region)
        
        # Initialize model
        logger.info("Initializing model...")
        model = WildfirePredictionModel(
            config=config,
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
            accumulation_steps=args.accumulation_steps,
            model_path=args.model_path,
            logger=logger
        )
        
        # Evaluate model
        logger.info("Evaluating model...")
        metrics = evaluate_model(model, data, logger)
        
        # Save evaluation metrics
        metrics_path = Path(args.model_path) / 'evaluation_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()