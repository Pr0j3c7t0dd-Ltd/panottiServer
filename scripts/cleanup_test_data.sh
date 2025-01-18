#!/bin/bash

# Script to clean up test data and logs
# WARNING: This will delete all contents in logs/ and data/ directories

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the parent directory (project root)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "Cleaning up test data and logs..."

# Clean logs directory
if [ -d "$PROJECT_ROOT/logs" ]; then
    echo "Removing contents of logs directory..."
    rm -rf "$PROJECT_ROOT/logs"/*
else
    echo "logs directory not found, skipping..."
fi

# Clean data directory
if [ -d "$PROJECT_ROOT/data" ]; then
    echo "Removing contents of data directory..."
    rm -rf "$PROJECT_ROOT/data"/*
else
    echo "data directory not found, skipping..."
fi

echo "Cleanup complete!" 