import ee
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import os
from pathlib import Path
import concurrent.futures
from tqdm import tqdm
import logging

class WildfireDataCollector:
    def __init__(self, config):
        """Initialize the data collector with configuration"""
        self.config = config
        self.regions = config['data'].processing_regions
        self.main_bounds = config['data'].default_region_bounds()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize Earth Engine
        try:
            ee.Initialize()
            self.sample_mode = False
            # Create geometry for full Mexico
            self.full_region = ee.Geometry.Rectangle([
                self.main_bounds['lon_min'],
                self.main_bounds['lat_min'],
                self.main_bounds['lon_max'],
                self.main_bounds['lat_max']
            ])
            # Create geometries for each region
            self.region_geometries = {
                name: ee.Geometry.Rectangle([
                    bounds['lon_min'],
                    bounds['lat_min'],
                    bounds['lon_max'],
                    bounds['lat_max']
                ])
                for name, bounds in self.regions.items()
            }
        except Exception as e:
            self.logger.error(f"Earth Engine initialization failed: {str(e)}")
            self.sample_mode = True

    def get_modis_data(self, start_date: str, end_date: str, region: ee.Geometry) -> ee.ImageCollection:
        """Fetch MODIS data for a specific region"""
        modis_collection = ee.ImageCollection('MODIS/006/MOD09GA') \
            .filterBounds(region) \
            .filterDate(start_date, end_date)
        
        def calculate_indices(image):
            ndvi = image.normalizedDifference(['sur_refl_b02', 'sur_refl_b01'])
            return image.addBands(ndvi.rename('NDVI'))
            
        return modis_collection.map(calculate_indices)

    def get_viirs_data(self, start_date: str, end_date: str, region: ee.Geometry) -> ee.ImageCollection:
        """Fetch VIIRS active fire data for a specific region"""
        return ee.ImageCollection('NOAA/VIIRS/001/VNP14A1') \
            .filterBounds(region) \
            .filterDate(start_date, end_date)

    def get_era5_weather(self, start_date: str, end_date: str, region: ee.Geometry) -> ee.ImageCollection:
        """Fetch ERA5 weather data for a specific region"""
        return ee.ImageCollection('ECMWF/ERA5/HOURLY') \
            .filterBounds(region) \
            .filterDate(start_date, end_date) \
            .select(self.config['data'].feature_columns)

    def collect_region_data(self, region_name: str, bounds: Dict, start_date: str, end_date: str) -> Dict:
        """Collect data for a specific region"""
        try:
            region_geometry = self.region_geometries[region_name]
            
            # Collect data for the region
            data = {
                'modis': self.get_modis_data(start_date, end_date, region_geometry),
                'viirs': self.get_viirs_data(start_date, end_date, region_geometry),
                'weather': self.get_era5_weather(start_date, end_date, region_geometry),
            }
            
            # Process and export data
            processed_data = self._process_region_data(data, region_name, bounds)
            
            return {
                'region': region_name,
                'data': processed_data
            }
            
        except Exception as e:
            self.logger.error(f"Error collecting data for region {region_name}: {str(e)}")
            return None

    def _process_region_data(self, data: Dict, region_name: str, bounds: Dict) -> Dict:
        """Process data for a specific region"""
        try:
            # Convert Earth Engine data to numpy arrays
            grid_size = self.config['data'].grid_size
            
            # Process each data type
            modis_data = self._ee_to_numpy(data['modis'], bounds, grid_size)
            viirs_data = self._ee_to_numpy(data['viirs'], bounds, grid_size)
            weather_data = self._ee_to_numpy(data['weather'], bounds, grid_size)
            
            return {
                'spatial': np.stack([modis_data, viirs_data], axis=-1),
                'temporal': weather_data,
                'region': region_name
            }
        except Exception as e:
            self.logger.error(f"Error processing region {region_name}: {str(e)}")
            return None

    def _ee_to_numpy(self, ee_object: ee.ImageCollection, bounds: Dict, grid_size: Tuple[int, int]) -> np.ndarray:
        """Convert Earth Engine object to numpy array with specific resolution"""
        try:
            # Get region coordinates
            region = [
                [bounds['lon_min'], bounds['lat_min']],
                [bounds['lon_min'], bounds['lat_max']],
                [bounds['lon_max'], bounds['lat_max']],
                [bounds['lon_max'], bounds['lat_min']]
            ]
            
            # Convert to array with specified resolution
            data = ee_object.getRegion(region, grid_size[0]).getInfo()
            return np.array(data).reshape(grid_size + (-1,))
            
        except Exception as e:
            self.logger.error(f"Error converting EE object to numpy: {str(e)}")
            return None

    def collect_all_data(self, start_date: str, end_date: str) -> Dict:
        """Collect data for all regions in parallel"""
        if not self.sample_mode:
            try:
                self.logger.info("Starting parallel data collection for all regions...")
                
                # Create output directories
                raw_dir = Path('data/raw')
                processed_dir = Path('data/processed')
                raw_dir.mkdir(parents=True, exist_ok=True)
                processed_dir.mkdir(parents=True, exist_ok=True)
                
                # Collect data for each region in parallel
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=self.config['data'].get('NUM_WORKERS', 4)
                ) as executor:
                    future_to_region = {
                        executor.submit(
                            self.collect_region_data, 
                            region_name, 
                            bounds,
                            start_date, 
                            end_date
                        ): region_name
                        for region_name, bounds in self.regions.items()
                    }
                    
                    # Process results as they complete
                    all_data = []
                    for future in tqdm(concurrent.futures.as_completed(future_to_region),
                                     total=len(self.regions),
                                     desc="Collecting regional data"):
                        region_name = future_to_region[future]
                        try:
                            data = future.result()
                            if data is not None:
                                all_data.append(data)
                        except Exception as e:
                            self.logger.error(f"Region {region_name} failed: {str(e)}")
                
                # Merge all regional data
                merged_data = self._merge_regional_data(all_data)
                
                # Save processed data
                np.savez(processed_dir / 'processed_data.npz', **merged_data)
                
                return merged_data
                
            except Exception as e:
                self.logger.error(f"Error in data collection: {str(e)}")
                return self._generate_sample_data()
        else:
            self.logger.info("Using sample data mode")
            return self._generate_sample_data()

    def _merge_regional_data(self, regional_data: List[Dict]) -> Dict:
        """Merge data from all regions"""
        if not regional_data:
            return self._generate_sample_data()
        
        # Combine data from all regions
        all_spatial = []
        all_temporal = []
        
        for region in regional_data:
            if region and 'data' in region:
                all_spatial.append(region['data']['spatial'])
                all_temporal.append(region['data']['temporal'])
        
        # Combine arrays
        return {
            'spatial': np.concatenate(all_spatial, axis=0),
            'temporal': np.concatenate(all_temporal, axis=0),
            'labels': self._generate_labels(all_spatial[0].shape[:-1])
        }

    def _generate_sample_data(self) -> Dict:
        """Generate sample data with new grid size"""
        grid_size = self.config['data'].grid_size
        return {
            'spatial': np.random.random((100,) + grid_size + (5,)),
            'temporal': np.random.random((100, 24, len(self.config['data'].feature_columns))),
            'labels': np.random.randint(0, 2, (100,) + grid_size)
        }

    def _generate_labels(self, shape: Tuple) -> np.ndarray:
        """Generate labels matching the data shape"""
        return np.random.randint(0, 2, shape)