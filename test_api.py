import base64
import requests

audio_path = "sample.mp3"  # same file you used before

with open(audio_path, "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

response = requests.post(
    "http://127.0.0.1:8000/detect",
    json={"audio_base64": audio_b64}
)

print(response.json())
