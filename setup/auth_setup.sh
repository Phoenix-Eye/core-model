#!/bin/bash

echo "🔐 Setting up Earth Engine authentication..."

# Check if credentials already exist
if [ -f "$HOME/.config/earthengine/credentials" ]; then
    echo "✅ Earth Engine credentials found"
    exit 0
fi

# Try to authenticate
python << EOF
import ee
try:
    ee.Authenticate()
    ee.Initialize()
    print("✅ Authentication successful!")
except Exception as e:
    print(f"⚠️  Authentication failed: {str(e)}")
    print("Will use sample data mode")
EOF