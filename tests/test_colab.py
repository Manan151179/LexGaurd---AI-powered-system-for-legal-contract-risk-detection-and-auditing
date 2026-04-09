import requests
import os
import time
from dotenv import load_dotenv

load_dotenv()
url = os.getenv("COLAB_API_URL")

# A real short prompt to see the raw output format
prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are LexGuard.<|eot_id|><|start_header_id|>user<|end_header_id|>

Say exactly: "The answer is: YES"<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

payload = {"text": prompt, "max_tokens": 30}

print(f"Testing API at: {url}")
start = time.time()
try:
    response = requests.post(url, json=payload, timeout=60)
    raw = response.json().get("response", "")
    print("=== RAW RESPONSE (repr) ===")
    print(repr(raw))
    print("=== RAW RESPONSE (print) ===")
    print(raw)
    print(f"\nTime: {time.time() - start:.2f}s")
except Exception as e:
    print(f"Error: {e}")
