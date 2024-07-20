#!/bin/bash

echo "🔍 Running pipeline verification..."

# Function to check if a directory has content
check_directory() {
    dir="$1"
    min_files="$2"
    
    if [ ! -d "$dir" ]; then
        echo "❌ Directory $dir does not exist"
        return 1
    fi
    
    file_count=$(find "$dir" -type f | wc -l)
    if [ "$file_count" -lt "$min_files" ]; then
        echo "❌ Directory $dir has insufficient files (found $file_count, expected at least $min_files)"
        return 1
    fi
    
    echo "✅ Directory $dir verified"
    return 0
}

# Function to check if Python imports work
check_imports() {
    python3 -c "
import tensorflow as tf
import tf_keras
import tensorflow_probability as tfp
from src.models.wildfire_model import WildfirePredictionModel
from src.utils.metrics import WildfireMetrics
from src.config import config
print('✅ All imports verified')
" 2>/dev/null

    if [ $? -ne 0 ]; then
        echo "❌ Import verification failed"
        return 1
    fi
    return 0
}

# Check directories
check_directory "data/processed" 1
check_directory "models/saved" 1
check_directory "logs" 1

# Check imports
check_imports

# Check if config is valid
python3 -c "from src.config import config; print('✅ Config verified')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Config verification failed"
    exit 1
fi

# Check if notebooks are valid
for nb in notebooks/*.ipynb; do
    jupyter nbconvert --to notebook --execute --inplace "$nb" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "❌ Notebook $nb verification failed"
        exit 1
    fi
done

echo "✅ All verifications passed"
exit 0