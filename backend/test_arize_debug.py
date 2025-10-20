"""
Debug script for Arize tracing
"""
import os
import logging
from dotenv import load_dotenv, find_dotenv

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG)

load_dotenv(find_dotenv())

print("🔍 Checking environment variables...")
print(f"OPENAI_API_KEY present: {bool(os.getenv('OPENAI_API_KEY'))}")

# Test with more verbose configuration
from arize.otel import register
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

print("\n🔧 Registering with Arize...")
print(f"Space ID: U3BhY2U6MjA1Ok9SZXY=")
print(f"API Key: ak-38963011-5f7d-44c7-9bb6-79405a02885c-iMsG0P-y9eaSauXCoxmEZPl3pM4omBh0")
print(f"Project: ai-trip-planner")

try:
    tracer_provider = register(
        space_id="U3BhY2U6MjA1Ok9SZXY=",
        api_key="ak-38963011-5f7d-44c7-9bb6-79405a02885c-iMsG0P-y9eaSauXCoxmEZPl3pM4omBh0",
        project_name="ai-trip-planner",
    )
    print("✅ Tracer provider registered successfully")
    
    # Also log to console to verify spans are created
    console_exporter = ConsoleSpanExporter()
    tracer_provider.add_span_processor(SimpleSpanProcessor(console_exporter))
    print("✅ Console span processor added")
    
except Exception as e:
    print(f"❌ Error registering tracer: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n🔧 Instrumenting OpenAI...")
from openinference.instrumentation.openai import OpenAIInstrumentor
try:
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
    print("✅ OpenAI instrumented successfully")
except Exception as e:
    print(f"❌ Error instrumenting OpenAI: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n🤖 Making OpenAI call...")
import openai
import time

try:
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Say 'hello' in one word."}],
    )
    print(f"✅ OpenAI Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"❌ Error calling OpenAI: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n⏳ Waiting for traces to export (5 seconds)...")
time.sleep(5)

print("\n✅ Test complete!")
print("📊 Check Arize at: https://app.arize.com")
print("💡 If no traces appear, there may be an authentication issue with the credentials.")

