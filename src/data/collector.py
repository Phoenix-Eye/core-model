import ee
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

class WildfireDataCollector:
    def __init__(self, region_bounds: Dict[str, float]):
        """
        Initialize the data collector with geographic bounds
        """
        self.bounds = region_bounds
        self.region = ee.Geometry.Rectangle([
            region_bounds['lon_min'],
            region_bounds['lat_min'],
            region_bounds['lon_max'],
            region_bounds['lat_max']
        ])
        ee.Initialize()
    
    def get_modis_data(self, start_date: str, end_date: str) -> ee.ImageCollection:
        """Fetch MODIS data"""
        modis_collection = ee.ImageCollection('MODIS/006/MOD09GA') \
            .filterBounds(self.region) \
            .filterDate(start_date, end_date)
        
        def calculate_indices(image):
            ndvi = image.normalizedDifference(['sur_refl_b02', 'sur_refl_b01'])
            return image.addBands(ndvi.rename('NDVI'))
            
        return modis_collection.map(calculate_indices)

    def get_viirs_data(self, start_date: str, end_date: str) -> ee.ImageCollection:
        """Fetch VIIRS active fire data"""
        return ee.ImageCollection('NOAA/VIIRS/001/VNP14A1') \
            .filterBounds(self.region) \
            .filterDate(start_date, end_date)
    
    def get_gedi_canopy(self) -> ee.Image:
        """Fetch GEDI canopy height data"""
        return ee.ImageCollection('LARSE/GEDI/GEDI02_A_002_MONTHLY') \
            .filterBounds(self.region) \
            .first() \
            .select('rh98')
    
    def get_era5_weather(self, start_date: str, end_date: str) -> ee.ImageCollection:
        """Fetch ERA5 weather data"""
        return ee.ImageCollection('ECMWF/ERA5/HOURLY') \
            .filterBounds(self.region) \
            .filterDate(start_date, end_date) \
            .select([
                'temperature_2m',
                'relative_humidity_2m',
                'u_component_of_wind_10m',
                'v_component_of_wind_10m',
                'total_precipitation'
            ])

    def get_terrain_data(self) -> ee.Image:
        """Fetch terrain data"""
        srtm = ee.Image('USGS/SRTMGL1_003')
        elevation = srtm.select('elevation')
        return ee.Terrain.products(elevation) \
            .select(['elevation', 'slope', 'aspect'])

    def get_landcover(self) -> ee.Image:
        """Fetch land cover classification"""
        return ee.ImageCollection('ESA/WorldCover/v100').first()

    def collect_all_data(self, start_date: str, end_date: str) -> Dict:
        """Collect all required data"""
        return {
            'modis': self.get_modis_data(start_date, end_date),
            'viirs': self.get_viirs_data(start_date, end_date),
            'gedi': self.get_gedi_canopy(),
            'weather': self.get_era5_weather(start_date, end_date),
            'terrain': self.get_terrain_data(),
            'landcover': self.get_landcover()
        }

    def export_to_drive(self, data: Dict, folder: str):
        """Export collected data to Google Drive"""
        for name, dataset in data.items():
            task = ee.batch.Export.image.toDrive(
                image=dataset.first() if isinstance(dataset, ee.ImageCollection) else dataset,
                description=f'export_{name}',
                folder=folder,
                scale=30,
                region=self.region
            )
            task.start()