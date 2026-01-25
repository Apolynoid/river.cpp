#!/bin/bash

# 1. Setup Directories
echo "Creating data directories..."
mkdir -p data

# 2. Download Benchmark Dataset (Covertype) - 75MB
if [ ! -f "data/covtype.data" ]; then
    echo "Downloading Covertype dataset (for Benchmarks)..."
    wget -q --show-progress https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz -O data/covtype.data.gz
    
    echo "Extracting Covertype..."
    gunzip -f data/covtype.data.gz
else
    echo "Covertype dataset already exists."
fi

echo "🎉 All datasets are ready!"
echo "   - Benchmark: data/covtype.data (581,012 samples)"
