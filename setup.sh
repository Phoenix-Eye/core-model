#!/bin/bash

# Create root directory and main subdirectories
mkdir -p {data/{raw,processed},models/saved,src/{data,models,utils},notebooks,scripts,tests}

# Create empty files
touch requirements.txt
touch README.md

# Create Python files
touch src/__init__.py
touch src/config.py
touch src/data/__init__.py
touch src/data/collector.py
touch src/data/processor.py
touch src/models/__init__.py
touch src/models/layers.py
touch src/models/losses.py
touch src/models/wildfire_model.py
touch src/utils/__init__.py
touch src/utils/metrics.py
touch tests/__init__.py

# Create notebook files
touch notebooks/data_exploration.ipynb
touch notebooks/model_evaluation.ipynb

# Create script files
touch scripts/train.py
touch scripts/predict.py

# Write requirements to requirements.txt
cat << 'EOF' > requirements.txt
tensorflow>=2.8.0
tensorflow-probability>=0.16.0
earthengine-api>=0.1.317
pandas>=1.4.0
numpy>=1.21.0
xarray>=0.20.0
geopandas>=0.10.0
rasterio>=1.2.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
jupyter>=1.0.0
pytest>=7.0.0
EOF

# Set appropriate permissions
chmod +x scripts/*.py

# Print success message
echo "Project structure created successfully!"