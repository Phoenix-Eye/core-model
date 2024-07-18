#!/bin/bash

# Create root directory and main subdirectories
mkdir -p phoenixeye/{data/{raw,processed},models/saved,src/{data,models,utils},notebooks,scripts,tests}

# Create empty files
touch phoenixeye/requirements.txt
touch phoenixeye/README.md

# Create Python files
touch phoenixeye/src/__init__.py
touch phoenixeye/src/config.py
touch phoenixeye/src/data/__init__.py
touch phoenixeye/src/data/collector.py
touch phoenixeye/src/data/processor.py
touch phoenixeye/src/models/__init__.py
touch phoenixeye/src/models/layers.py
touch phoenixeye/src/models/losses.py
touch phoenixeye/src/models/wildfire_model.py
touch phoenixeye/src/utils/__init__.py
touch phoenixeye/src/utils/metrics.py
touch phoenixeye/tests/__init__.py

# Create notebook files
touch phoenixeye/notebooks/data_exploration.ipynb
touch phoenixeye/notebooks/model_evaluation.ipynb

# Create script files
touch phoenixeye/scripts/train.py
touch phoenixeye/scripts/predict.py

# Write requirements to requirements.txt
cat << 'EOF' > phoenixeye/requirements.txt
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
chmod +x phoenixeye/scripts/*.py

# Print success message
echo "Project structure created successfully!"