#!/bin/bash

# Kill any existing backend processes
echo "🔄 Stopping any existing backend processes..."
pkill -f "python3 main.py" 2>/dev/null || true

# Wait a moment for processes to fully terminate
sleep 2

# Start the backend
echo "🚀 Starting backend on port 8002..."
python3 main.py
