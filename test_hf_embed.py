import requests
import os
from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
model_id = "sentence-transformers/all-mpnet-base-v2"
api_url = f"https://api-inference.huggingface.co/models/{model_id}"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

def query(payload):
	response = requests.post(api_url, headers=headers, json=payload)
	return response.json()

output = query({
	"inputs": "Test sentence",
    "options": {"wait_for_model": True}
})

import json
print(json.dumps(output, indent=2))
if isinstance(output, list) and len(output) > 0:
    print(f"Dimension: {len(output)}")
