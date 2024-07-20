#!/bin/bash

echo "🌍 Checking Earth Engine registration status..."

# Function to check if user wants to register
ask_for_registration() {
    echo "Earth Engine access is not set up. Would you like to:"
    echo "1) Register for Earth Engine access (requires Google Account)"
    echo "2) Continue with sample data mode"
    read -p "Enter choice [1/2]: " choice
    
    case $choice in
        1)
            echo "📝 Please follow these steps:"
            echo "1. Visit https://developers.google.com/earth-engine/guides/access"
            echo "2. Sign in with your Google Account"
            echo "3. Click 'Sign Up' button"
            echo "4. Complete the registration form"
            echo "5. Wait for approval email (usually within 24 hours)"
            echo ""
            read -p "Press Enter once you've completed registration or Ctrl+C to cancel"
            return 0
            ;;
        2)
            echo "✨ Continuing with sample data mode..."
            return 1
            ;;
        *)
            echo "❌ Invalid choice. Defaulting to sample data mode..."
            return 1
            ;;
    esac
}

# Check Earth Engine registration status
python << EOF
import ee
try:
    ee.Authenticate()
    ee.Initialize(project='donativehub')
    print("✅ Earth Engine access verified!")
    exit(0)
except Exception as e:
    if "Not signed up for Earth Engine" in str(e):
        print("⚠️  Earth Engine access not set up")
        exit(1)
    else:
        print(f"⚠️  Other Earth Engine error: {str(e)}")
        exit(2)
EOF

exit_code=$?

case $exit_code in
    0)
        echo "✅ Earth Engine is properly configured"
        ;;
    1)
        echo "ℹ️  Earth Engine registration required"
        if ask_for_registration; then
            echo "🔄 Please run this script again after receiving approval email"
            exit 1
        fi
        ;;
    2)
        echo "⚠️  Using sample data mode due to Earth Engine error"
        ;;
esac