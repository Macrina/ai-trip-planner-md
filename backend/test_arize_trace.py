"""
Test script to verify Arize OpenAI instrumentation
"""
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Arize instrumentation setup
from arize.otel import register

tracer_provider = register(
    space_id=os.getenv("ARIZE_SPACE_ID", "U3BhY2U6MjA1Ok9SZXY="),
    api_key=os.getenv("ARIZE_API_KEY", "ak-38963011-5f7d-44c7-9bb6-79405a02885c-iMsG0P-y9eaSauXCoxmEZPl3pM4omBh0"),
    project_name=os.getenv("ARIZE_PROJECT_NAME", "ai-trip-planner"),
)

from openinference.instrumentation.openai import OpenAIInstrumentor
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

# Now make OpenAI call
import openai

client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-3.5-turbo",  # Changed from gpt-5 (doesn't exist) to gpt-3.5-turbo
    messages=[{"role": "user", "content": "Write a haiku about observability."}],
)

print("\n✅ OpenAI Response:")
print(response.choices[0].message.content)
print("\n✅ Trace sent to Arize! Check your Arize dashboard.")
print("📊 View traces at: https://app.arize.com")

# Force flush to ensure traces are sent
import time
time.sleep(2)  # Give time for traces to export

