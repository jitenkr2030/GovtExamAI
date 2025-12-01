#!/bin/bash
echo "Starting Government Exam AI API Service..."

# Install dependencies
pip install -r requirements.txt

# Load model (if exists)
if [ -f "/workspace/models/trained_model.pt" ]; then
    echo "Loading trained model..."
else
    echo "Using default model configuration..."
fi

# Start API server
python main.py
