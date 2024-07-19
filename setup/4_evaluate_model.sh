#!/bin/bash

echo "📊 Starting model evaluation..."

# Activate virtual environment
source venv/bin/activate

echo "📈 Running evaluation notebooks..."
jupyter nbconvert --to notebook --execute notebooks/model_evaluation.ipynb --output-dir logs
jupyter nbconvert --to html notebooks/model_evaluation.ipynb --output-dir logs

echo "✅ Model evaluation complete!"