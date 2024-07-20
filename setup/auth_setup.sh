#!/bin/bash

echo "🔐 Setting up Earth Engine authentication..."

# First check registration
./setup/check_ee_registration.sh
registration_status=$?

if [ $registration_status -eq 0 ]; then
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
else
    echo "📝 Using sample data mode (Earth Engine registration pending)"
    
    # Create a flag file to indicate we're in sample mode
    touch .using_sample_data
fi