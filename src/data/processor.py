import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict
import xarray as xr

class WildfireDataProcessor:
    def __init__(self, config: Dict):
        self.config = config
        self.scalers = {
            'spatial': StandardScaler(),
            'tabular': StandardScaler()
        }

    def align_spatial_data(self, data: Dict) -> np.ndarray:
        """Align and resample spatial data to common grid"""
        grid_size = self.config['data'].grid_size
        aligned_data = []
        
        for dataset in data.values():
            # Resample to common grid
            resampled = dataset.resample(grid_size)
            aligned_data.append(resampled)
            
        return np.stack(aligned_data, axis=-1)

    def process_tabular_features(self, data: Dict) -> np.ndarray:
        """Process tabular features"""
        features = []
        for feature in self.config['data'].feature_columns:
            if feature in data['weather']:
                features.append(data['weather'][feature])
                
        tabular_data = np.stack(features, axis=-1)
        return self.scalers['tabular'].fit_transform(tabular_data)

    def create_sequences(self, data: np.ndarray, sequence_length: int) -> np.ndarray:
        """Create sequences for temporal analysis"""
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:i + sequence_length])
        return np.array(sequences)

    def prepare_data(self, raw_data: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for model training"""
        # Process spatial data
        spatial_data = self.align_spatial_data(raw_data)
        spatial_data = self.scalers['spatial'].fit_transform(
            spatial_data.reshape(-1, spatial_data.shape[-1])
        ).reshape(spatial_data.shape)
        
        # Process tabular data
        tabular_data = self.process_tabular_features(raw_data)
        
        # Create sequences
        X_spatial = self.create_sequences(
            spatial_data, 
            self.config['data'].time_steps
        )
        X_tabular = self.create_sequences(
            tabular_data, 
            self.config['data'].time_steps
        )
        
        # Create target variable (fire/no-fire)
        y = np.where(raw_data['viirs'].values > 0, 1, 0)
        y = self.create_sequences(y, self.config['data'].time_steps)
        
        return X_spatial, X_tabular, y

    def inverse_transform(self, data: np.ndarray, data_type: str) -> np.ndarray:
        """Inverse transform scaled data"""
        return self.scalers[data_type].inverse_transform(data)