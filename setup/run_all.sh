#!/bin/bash

# Function to check for errors and apply fixes
check_and_fix() {
    error_type="$1"
    error_message="$2"
    
    case $error_type in
        "tf_keras")
            echo "❌ TF-Keras issue detected. Running environment fix..."
            ./setup/fix_environment.sh
            ;;
        "notebook")
            echo "❌ Notebook format issue detected. Running notebook fix..."
            python setup/fix_notebooks.py
            ;;
        "config")
            echo "❌ Config issue detected. Running config fix..."
            ./setup/fix_environment.sh
            ;;
        *)
            echo "❌ Unknown error: $error_message"
            exit 1
            ;;
    esac
}

# Function to analyze error output
analyze_error() {
    error_output="$1"
    
    if echo "$error_output" | grep -q "No module named 'tf_keras'"; then
        check_and_fix "tf_keras"
        return 0
    elif echo "$error_output" | grep -q "outputs.*required property"; then
        check_and_fix "notebook"
        return 0
    elif echo "$error_output" | grep -q "mutable default.*dict.*field"; then
        check_and_fix "config"
        return 0
    else
        return 1
    fi
}

echo "🚀 Running complete Phoenix Eye pipeline..."

# Ensure environment is properly set up
source venv/bin/activate

# Check for required dependencies
if ! pip show tf-keras > /dev/null; then
    echo "❌ Missing required dependencies. Running environment fix..."
    ./setup/fix_environment.sh
fi

# Create temporary directory for error logs
mkdir -p logs/temp

# Run each step in sequence with error checking and automatic fixes
for step in {1..5}; do
    script="./setup/${step}_*.sh"
    echo "Running step ${step}..."
    
    # Try running the script up to 3 times with fixes
    for attempt in {1..3}; do
        # Run script and capture output
        error_log="logs/temp/step_${step}_attempt_${attempt}.log"
        if output=$(bash $script 2>&1); then
            echo "✅ Step ${step} completed successfully!"
            break
        else
            echo "$output" > "$error_log"
            echo "❌ Step ${step} failed (attempt $attempt). Analyzing error..."
            
            if analyze_error "$output"; then
                echo "🔄 Retrying step ${step} after applying fix..."
                continue
            else
                echo "❌ Unrecoverable error in step ${step}. Please check logs at $error_log"
                exit 1
            fi
        fi
    done
    
    # If we've tried 3 times and still failed, exit
    if [ $attempt -eq 3 ]; then
        echo "❌ Step ${step} failed after 3 attempts. Please check the logs."
        exit 1
    fi
done

# Cleanup temporary logs
rm -rf logs/temp

echo "✅ Pipeline complete!"

# Run final verification
echo "🔍 Running final verification..."
./setup/verify_pipeline.sh

if [ $? -eq 0 ]; then
    echo "🎉 Pipeline completed and verified successfully!"
else
    echo "⚠️ Pipeline completed but verification found issues. Please check the logs."
    exit 1
fi