# config.py
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

@dataclass
class DataConfig:
    @staticmethod
    def default_region_bounds():
        return {
            'lat_min': 14.5321,  
            'lat_max': 32.7185, 
            'lon_min': -118.4079,
            'lon_max': -86.7132,
        }

    @staticmethod
    def processing_regions():
        return {
            'northwest': {
                'lat_min': 23.0,
                'lat_max': 32.7185,
                'lon_min': -118.4079,
                'lon_max': -105.0
            },
            'northeast': {
                'lat_min': 23.0,
                'lat_max': 32.7185,
                'lon_min': -105.0,
                'lon_max': -97.0
            },
            'central': {
                'lat_min': 18.0,
                'lat_max': 23.0,
                'lon_min': -105.0,
                'lon_max': -95.0
            },
            'southeast': {
                'lat_min': 14.5321,
                'lat_max': 23.0,
                'lon_min': -95.0,
                'lon_max': -86.7132
            }
        }

    region_bounds: Dict[str, float] = field(default_factory=default_region_bounds)
    processing_regions: Dict[str, Dict[str, float]] = field(default_factory=processing_regions)
    grid_size: Tuple[int, int] = (256, 256)  # Increased for better resolution
    time_steps: int = 24  # Increased for better temporal resolution
    channels: int = 5
    feature_columns: List[str] = field(default_factory=lambda: [
        'temperature_2m',
        'relative_humidity_2m',
        'u_component_of_wind_10m',
        'v_component_of_wind_10m',
        'total_precipitation'
    ])
    use_gpu: bool = True
    precision: str = 'mixed_float16'  # Use mixed precision for better GPU performance

@dataclass
class ModelConfig:
    input_shape: Tuple[int, int, int, int] = field(default_factory=lambda: (None, 256, 256, 5))
    num_features: int = 10
    num_ensemble: int = 3
    spatial_weight: float = 0.6
    temporal_weight: float = 0.4
    learning_rate: float = 0.001
    batch_size: int = 16
    epochs: int = 200
    validation_split: float = 0.2
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 4

@dataclass
class TrainingConfig:
    early_stopping_patience: int = 15  
    reduce_lr_patience: int = 8
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-6
    model_save_path: str = "models/saved/"
    checkpoint_path: str = "models/saved/checkpoints/"
    log_dir: str = "logs/"
    use_amp: bool = True 
    gradient_clip_norm: float = 1.0
    warmup_epochs: int = 5
    profile_batch: int = 100 

config = {
    'data': DataConfig(),
    'model': ModelConfig(),
    'training': TrainingConfig()
}