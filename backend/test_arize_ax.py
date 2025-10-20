#!/usr/bin/env python3
"""
Test script for Arize AX observability instrumentation
Run this to verify Arize AX tracing is working correctly
"""

import os
import time
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# Arize AX setup
try:
    from arize.otel import register
    from openinference.instrumentation.openai import OpenAIInstrumentor
    from openinference.instrumentation.langchain import LangChainInstrumentor
    
    # Arize AX configuration
    arize_space_id = os.getenv("ARIZE_SPACE_ID")
    arize_api_key = os.getenv("ARIZE_API_KEY")
    arize_project_name = os.getenv("ARIZE_PROJECT_NAME", "ai-trip-planner")
    
    print(f"🔭 Setting up Arize AX observability...")
    print(f"📊 Project: {arize_project_name}")
    print(f"🔑 Space ID: {arize_space_id[:10] if arize_space_id else 'Not set'}...")
    
    if not arize_space_id or not arize_api_key:
        print("❌ Arize AX credentials not found!")
        print("💡 Set ARIZE_SPACE_ID and ARIZE_API_KEY environment variables")
        exit(1)
    
    # Register Arize AX tracer provider
    tracer_provider = register(
        space_id=arize_space_id,
        api_key=arize_api_key,
        project_name=arize_project_name,
    )
    
    # Instrument OpenAI and LangChain
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    
    print("✅ Arize AX instrumentation enabled")
    
    # Test OpenAI call with tracing
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    print("\n🧪 Testing OpenAI call with Arize AX tracing...")
    
    def test_openai_call():
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Hello! This is a test call for Arize AX tracing."}
            ],
            max_tokens=50
        )
        return response.choices[0].message.content
    
    # Make the test call
    result = test_openai_call()
    print(f"🤖 OpenAI response: {result}")
    
    # Wait for traces to be exported
    print("\n⏳ Waiting for traces to be exported to Arize AX...")
    time.sleep(3)
    
    print(f"\n✅ Test completed!")
    print(f"📊 Check your Arize AX dashboard at: https://app.arize.com")
    print(f"🔍 Look for project: {arize_project_name}")
    
except ImportError as e:
    print(f"❌ Arize not installed: {e}")
    print("💡 Install with: pip install arize openinference-instrumentation-openai openinference-instrumentation-langchain")
except Exception as e:
    print(f"❌ Arize AX setup failed: {e}")
    print("💡 Check your Arize AX credentials and network connection")
