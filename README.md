# Phoenix Eye: AI-Powered Wildfire Prediction Model

An advanced machine learning system designed to forecast the spread of wildfires using a dual-model approach combining Convolutional Neural Networks (CNN) + LSTM and Random Forest algorithms.

## Models Overview

### 1. CNN + LSTM Model

#### Architecture

- Combines Convolutional Neural Networks (CNN) and Long Short Term Memory (LSTM) Neural Networks
- Trained using Adam optimization algorithm
- Binary cross-entropy loss function
- Training: 20 epochs with batch size of 5
- Implements early stopping and learning rate reduction

#### Input Data

- Sequences of 5 64x64 grayscale images
- Images represent geospatial and temporal data
- Data normalized and split: 80% training, 20% validation

#### Performance Metrics

| Metric | Value |
|--------|--------|
| Fire Precision | 0.92 |
| Fire Recall | 0.75 |
| No-Fire Precision | 0.99 |
| No-Fire Recall | 0.99 |

### 2. Random Forest Model

#### Dataset Characteristics

- 18,445 samples
- Each sample represents 64km x 64km region
- Variables include:
  - Topography
  - Weather conditions
  - Vegetation data
  - Previous day's fire mask
  - Current day's fire mask

#### Performance Metrics

| Metric | Value |
|--------|--------|
| Fire Precision | 0.30 |
| Fire Recall | 0.44 |
| No-Fire Precision | 0.98 |
| No-Fire Recall | 0.97 |

## Data Sources

Primary data sources used for model training:

- **MODIS**: Moderate Resolution Imaging Spectroradiometer data
- **Meteomatics API**: Weather and environmental data
- **FIRMS**: Active fire data from NASA
- **Copernicus DEM**: Digital Surface Model data

## Model Training Instructions

### CNN + LSTM Model

1. Execute notebook in Google Colab
2. Load required datasets:
   - Fire occurrence data
   - Climate data

### Random Forest Model

1. Download dataset from [Kaggle Wildfire Spread Dataset](https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread)
2. Follow notebook instructions for model training

## Future Improvements

1. **AutoML Pipelines**
   - Deploy AutoML for improved model architecture
   - Optimize hyperparameter testing

2. **Data Expansion**
   - Increase dataset size
   - Add more diverse geographical regions

3. **Model Enhancement**
   - Experiment with deeper architectures
   - Add more layers to capture complex features

4. **Production Deployment**
   - Deploy model in production environment
   - Implement real-time prediction capabilities

## Technologies Used

- Python 3
- Keras/TensorFlow
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
