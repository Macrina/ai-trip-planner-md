#!/bin/bash
# Setup script for Arize AX observability

echo "🔭 Setting up Arize AX observability for AI Trip Planner..."

# Install Arize AX packages
echo "📦 Installing Arize AX packages..."
pip install arize openinference-instrumentation-openai openinference-instrumentation-langchain

# Check environment variables
echo "🔍 Checking Arize AX credentials..."
if [ -z "$ARIZE_SPACE_ID" ]; then
    echo "❌ ARIZE_SPACE_ID not set!"
    echo "💡 Set it with: export ARIZE_SPACE_ID='your-space-id'"
fi

if [ -z "$ARIZE_API_KEY" ]; then
    echo "❌ ARIZE_API_KEY not set!"
    echo "💡 Set it with: export ARIZE_API_KEY='your-api-key'"
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ OPENAI_API_KEY not set!"
    echo "💡 Set it with: export OPENAI_API_KEY='your-openai-key'"
fi

echo "✅ Setup complete!"
echo "🧪 Test with: python test_arize_ax.py"
echo "🚀 Start app: python main.py"
echo "📊 Dashboard: https://app.arize.com"
