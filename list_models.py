#!/usr/bin/env python3
"""List available Gemini models for your API key."""
import requests
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("ERROR: GEMINI_API_KEY not found in .env")
    exit(1)

url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"

print("Fetching available models...")
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    print("\n=== Available Models ===\n")
    for model in data.get('models', []):
        name = model.get('name', 'N/A')
        methods = model.get('supportedGenerationMethods', [])
        print(f"  {name}")
        print(f"    Methods: {', '.join(methods)}")
        print()
else:
    print(f"Error: {response.status_code}")
    print(response.text)
