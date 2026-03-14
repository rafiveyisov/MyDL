#!/bin/bash

# RNN vs LSTM Sentiment Analyzer - Quick Start Script

echo "=================================="
echo "RNN vs LSTM Sentiment Analyzer"
echo "Quick Start Setup"
echo "=================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
echo ""

# Train models
echo "=================================="
echo "STEP 1: Training Models"
echo "This may take 15-30 minutes..."
echo "=================================="
python3 train_models.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Models trained successfully!"
    echo ""
    echo "=================================="
    echo "STEP 2: Starting Server"
    echo "=================================="
    python3 app.py
else
    echo ""
    echo "✗ Model training failed. Please check the errors above."
    exit 1
fi
