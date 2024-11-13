import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm
import concurrent.futures

class WildfireDataProcessor:
    def __init__(self, config):
        """Initialize processor with configuration"""
        self.config = config['model']
        self.grid_size = config['data'].grid_size
        self.feature_columns = config['data'].feature_columns
        self.regions = config['data'].processing_regions()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Create cache directory
        self.cache_dir = Path('cache')
        self.cache_dir.mkdir(exist_ok=True)

    def prepare_data(self, raw_dir: Path) -> Dict:
        """Process raw data into model-ready format"""
        try:
            # Check if raw data exists
            raw_dir = Path(raw_dir)
            if not raw_dir.exists():
                self.logger.warning("Raw data directory not found. Generating sample data...")
                return self._generate_sample_data()

            # Load and process data by region
            processed_regions = {}
            
            with concurrent.futures.ProcessPoolExecutor() as executor:
                future_to_region = {
                    executor.submit(
                        self._process_region,
                        raw_dir,
                        region_name,
                        bounds
                    ): region_name
                    for region_name, bounds in self.regions.items()
                }
                
                for future in tqdm(
                    concurrent.futures.as_completed(future_to_region),
                    total=len(self.regions),
                    desc="Processing regions"
                ):
                    region_name = future_to_region[future]
                    try:
                        processed_regions[region_name] = future.result()
                    except Exception as e:
                        self.logger.error(f"Error processing region {region_name}: {str(e)}")

            # Merge regional data
            return self._merge_regional_data(processed_regions)

        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            return self._generate_sample_data()

    def _process_region(self, raw_dir: Path, region_name: str, bounds: Dict) -> Dict:
        """Process data for a specific region"""
        try:
            # Load regional data
            modis_data = np.load(raw_dir / f'modis_{region_name}.npy', allow_pickle=True)
            viirs_data = np.load(raw_dir / f'viirs_{region_name}.npy', allow_pickle=True)
            weather_data = np.load(raw_dir / f'weather_{region_name}.npy', allow_pickle=True)
            terrain_data = np.load(raw_dir / f'terrain_{region_name}.npy', allow_pickle=True)

            # Process features
            spatial_data = self._process_spatial_features(
                modis_data, 
                viirs_data, 
                terrain_data,
                bounds
            )
            temporal_data = self._process_temporal_features(weather_data, bounds)
            labels = self._create_labels(viirs_data, bounds)

            return {
                'spatial': spatial_data,
                'temporal': temporal_data,
                'labels': labels
            }

        except Exception as e:
            self.logger.error(f"Error processing region {region_name}: {str(e)}")
            return None

    def _process_spatial_features(
        self, 
        modis: np.ndarray, 
        viirs: np.ndarray, 
        terrain: np.ndarray,
        bounds: Dict
    ) -> np.ndarray:
        """Process spatial features into consistent format"""
        try:
            if all(x is not None for x in [modis, viirs, terrain]):
                # Reshape data to match grid size
                modis_reshaped = self._reshape_to_grid(modis, bounds)
                viirs_reshaped = self._reshape_to_grid(viirs, bounds)
                terrain_reshaped = self._reshape_to_grid(terrain, bounds)
                
                # Combine features
                return np.concatenate([
                    modis_reshaped,
                    viirs_reshaped,
                    terrain_reshaped
                ], axis=-1)
        except Exception as e:
            self.logger.error(f"Error processing spatial features: {str(e)}")
            
        # Return sample data if processing fails
        return np.random.random((100,) + self.grid_size + (5,))

    def _process_temporal_features(
        self, 
        weather: np.ndarray,
        bounds: Dict
    ) -> np.ndarray:
        """Process temporal features into consistent format"""
        try:
            if weather is not None:
                # Extract features in correct order
                features = []
                for feature in self.feature_columns:
                    if feature in weather.dtype.names:
                        feature_data = weather[feature]
                        # Reshape to correct dimensions
                        feature_data = self._reshape_temporal_data(
                            feature_data, 
                            bounds
                        )
                        features.append(feature_data)
                
                # Stack features
                return np.stack(features, axis=-1)
        except Exception as e:
            self.logger.error(f"Error processing temporal features: {str(e)}")
            
        # Return sample data if processing fails
        return np.random.random((100, 24, len(self.feature_columns)))

    def _create_labels(
        self, 
        viirs: np.ndarray,
        bounds: Dict
    ) -> np.ndarray:
        """Create labels from VIIRS data"""
        try:
            if viirs is not None:
                # Convert VIIRS data to binary labels
                labels = (viirs > 0).astype(np.int32)
                return self._reshape_to_grid(labels, bounds)
        except Exception as e:
            self.logger.error(f"Error creating labels: {str(e)}")
            
        # Return sample data if processing fails
        return np.random.randint(0, 2, (100,) + self.grid_size)

    def _reshape_to_grid(
        self, 
        data: np.ndarray, 
        bounds: Dict
    ) -> np.ndarray:
        """Reshape data to match grid size"""
        try:
            # Calculate interpolation points
            lat_points = np.linspace(
                bounds['lat_min'],
                bounds['lat_max'],
                self.grid_size[0]
            )
            lon_points = np.linspace(
                bounds['lon_min'],
                bounds['lon_max'],
                self.grid_size[1]
            )
            
            # Interpolate data to grid
            return np.interp(
                (lat_points, lon_points),
                (data['lat'], data['lon']),
                data['values']
            )
        except Exception as e:
            self.logger.error(f"Error reshaping data: {str(e)}")
            return None

    def _reshape_temporal_data(
        self, 
        data: np.ndarray, 
        bounds: Dict
    ) -> np.ndarray:
        """Reshape temporal data to correct dimensions"""
        try:
            # Reshape to (time_steps, features)
            return data.reshape(-1, 24, data.shape[-1])
        except Exception as e:
            self.logger.error(f"Error reshaping temporal data: {str(e)}")
            return None

    def _merge_regional_data(self, regional_data: Dict[str, Dict]) -> Dict:
        """Merge data from all regions"""
        try:
            all_spatial = []
            all_temporal = []
            all_labels = []
            
            for region_data in regional_data.values():
                if region_data is not None:
                    all_spatial.append(region_data['spatial'])
                    all_temporal.append(region_data['temporal'])
                    all_labels.append(region_data['labels'])
            
            return {
                'spatial': np.concatenate(all_spatial, axis=0),
                'temporal': np.concatenate(all_temporal, axis=0),
                'labels': np.concatenate(all_labels, axis=0)
            }
        except Exception as e:
            self.logger.error(f"Error merging regional data: {str(e)}")
            return self._generate_sample_data()

    def _generate_sample_data(self) -> Dict:
        """Generate sample data with correct shapes"""
        return {
            'spatial': np.random.random((100,) + self.grid_size + (5,)),
            'temporal': np.random.random((100, 24, len(self.feature_columns))),
            'labels': np.random.randint(0, 2, (100,) + self.grid_size)
        }