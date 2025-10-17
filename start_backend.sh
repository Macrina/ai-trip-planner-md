#!/bin/bash

# AI Trip Planner - Start Backend with Arize (Python 3.12)

echo "🛑 Stopping any running backends..."
killall -9 Python python python3 2>/dev/null
sleep 2

echo "🚀 Starting backend with Python 3.12 (Arize enabled) on port 8000..."

# Navigate to project root
cd "$(dirname "$0")"

# Activate Python 3.12 virtual environment
source venv312/bin/activate

# Navigate to backend directory
cd backend

# Start the backend
echo ""
echo "✅ Backend starting..."
echo "📍 URL: http://localhost:8000"
echo "🔭 Arize Tracing: ENABLED"
echo ""

# Run the backend (foreground)
python main.py

