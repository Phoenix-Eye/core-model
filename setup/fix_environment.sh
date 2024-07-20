#!/bin/bash

echo "🔧 Fixing environment setup..."

# Deactivate virtual environment if it exists
deactivate 2>/dev/null || true

# Remove old virtual environment
rm -rf venv

# Create new virtual environment
python -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies with specific order to avoid conflicts
pip install tensorflow
pip install tf-keras
pip install 'tensorflow-probability[tf]'

# Install remaining requirements
pip install -r requirements.txt

# Fix notebook format issues
jupyter nbformat --to=notebook --validate notebooks/data_exploration.ipynb
jupyter nbformat --to=notebook --validate notebooks/model_evaluation.ipynb

echo "✅ Environment fixed successfully!"