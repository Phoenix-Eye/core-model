from dataclasses import dataclass, field
from typing import Dict, List, Tuple

@dataclass
class DataConfig:
    @staticmethod
    def default_region_bounds():
        return {
            'lat_min': 30.73819440764155,
            'lat_max': 31.31219440764155,
            'lon_min': -111.2942054407774,
            'lon_max': -110.6342054407774
        }

    region_bounds: Dict[str, float] = field(default_factory=default_region_bounds)
    grid_size: Tuple[int, int] = (64, 64)
    time_steps: int = 5
    channels: int = 5
    feature_columns: List[str] = field(default_factory=lambda: [
        'temperature_2m',
        'relative_humidity_2m',
        'u_component_of_wind_10m',
        'v_component_of_wind_10m',
        'total_precipitation'
    ])

@dataclass
class ModelConfig:
    input_shape: Tuple[int, int, int, int] = field(default_factory=lambda: (None, 64, 64, 5))
    num_features: int = 10
    num_ensemble: int = 3
    spatial_weight: float = 0.6
    temporal_weight: float = 0.4
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2

@dataclass
class TrainingConfig:
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-6
    model_save_path: str = "models/saved/"
    checkpoint_path: str = "models/saved/checkpoints/"
    log_dir: str = "logs/"

config = {
    'data': DataConfig(),
    'model': ModelConfig(),
    'training': TrainingConfig()
}