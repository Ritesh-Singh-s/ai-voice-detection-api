import base64
import requests

API_KEY = "SECRET_API_KEY_123"   # must match api.py
API_URL = "http://127.0.0.1:8000/detect"

audio_path = "sample.mp3"  # or sample.mp3

with open(audio_path, "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

response = requests.post(
    API_URL,
    headers={
        "x-api-key": API_KEY
    },
    json={
        "audio_base64": audio_b64
    }
)

print(response.json())
