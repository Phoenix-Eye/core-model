import os
import sys
import argparse
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

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
    parser = argparse.ArgumentParser(description='Make wildfire predictions')
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to saved model directory'
    )
    parser.add_argument(
        '--num-models',
        type=int,
        default=3,
        help='Number of ensemble models'
    )
    parser.add_argument(
        '--region',
        type=str,
        default='nogales',
        help='Region for prediction'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        required=True,
        help='Path to save predictions'
    )
    return parser.parse_args()

def load_region_bounds(region: str) -> dict:
    """Load geographic bounds for specified region"""
    region_bounds = {
        'nogales': {
            'lat_min': 30.73819440764155,
            'lat_max': 31.31219440764155,
            'lon_min': -111.2942054407774,
            'lon_max': -110.6342054407774
        }
        # Add other regions as needed
    }
    
    if region not in region_bounds:
        raise ValueError(f"Unknown region: {region}")
    
    return region_bounds[region]

def make_predictions(model, data, threshold: float = 0.5):
    """Make predictions with uncertainty estimation"""
    predictions, uncertainties = model.predict(
        x=[data['spatial'], data['temporal']],
        return_uncertainty=True
    )
    
    # Apply threshold to get binary predictions
    binary_predictions = (predictions > threshold).astype(int)
    
    return {
        'predictions': predictions,
        'binary_predictions': binary_predictions,
        'uncertainties': uncertainties
    }

def save_predictions(predictions: dict, output_path: str, region: str):
    """Save predictions and uncertainties"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(output_path) / region / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as numpy arrays
    np.save(output_dir / 'predictions.npy', predictions['predictions'])
    np.save(output_dir / 'binary_predictions.npy', predictions['binary_predictions'])
    np.save(output_dir / 'uncertainties.npy', predictions['uncertainties'])
    
    # Create metadata file
    metadata = {
        'timestamp': timestamp,
        'region': region,
        'mean_confidence': float(np.mean(1 - predictions['uncertainties'])),
        'threshold': 0.5
    }
    
    pd.DataFrame([metadata]).to_csv(output_dir / 'metadata.csv', index=False)
    
    return output_dir

def main():
    args = parse_args()
    
    # Setup logging
    log_path = Path('logs') / f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logging(log_path)
    
    try:
        # Load region configuration
        region_bounds = load_region_bounds(args.region)
        
        # Initialize data collector and processor
        logger.info("Initializing data collection...")
        collector = WildfireDataCollector(region_bounds)
        processor = WildfireDataProcessor(config)
        
        # Collect latest data
        logger.info("Collecting current data...")
        current_date = datetime.now().strftime('%Y-%m-%d')
        raw_data = collector.collect_all_data(
            start_date=current_date,
            end_date=current_date
        )
        
        # Process data
        logger.info("Processing data...")
        processed_data = processor.prepare_data(raw_data)
        
        # Initialize and load model
        logger.info("Loading model...")
        model = WildfirePredictionModel(
            config=config['model'],
            num_ensemble=args.num_models
        )
        model.load_models(args.model_path, args.num_models)
        
        # Make predictions
        logger.info("Making predictions...")
        predictions = make_predictions(model, processed_data)
        
        # Save results
        logger.info("Saving predictions...")
        output_dir = save_predictions(predictions, args.output_path, args.region)
        logger.info(f"Predictions saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        raise
        
if __name__ == '__main__':
    main()