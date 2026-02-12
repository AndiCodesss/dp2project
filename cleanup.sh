#!/bin/bash

# Cleanup script for DP2 project
# Make executable: chmod +x cleanup.sh
# Usage: ./cleanup.sh [--hard]
# Usage: ./cleanup.sh [--hard]
# Default: removes output/ directory only
# --hard: removes output/ AND all dataset files

echo "🧹 Cleaning up..."

# Always clean output directory
if [ -d "output" ]; then
    rm -rf output
    echo "✓ Removed output/"
fi

# Hard cleanup: remove all datasets and generated files
if [ "$1" = "--hard" ] || [ "$1" = "-h" ]; then
    echo ""
    echo "💥 Hard cleanup mode - removing all datasets!"
    
    [ -d "crypto_data_4h" ] && rm -rf crypto_data_4h && echo "✓ Removed crypto_data_4h/"
    [ -d "crypto_data_parquet" ] && rm -rf crypto_data_parquet && echo "✓ Removed crypto_data_parquet/"
    
    echo "⚠️  All datasets removed. Run notebook 01 to regenerate data."
fi

echo ""
echo "✨ Cleanup complete!"
