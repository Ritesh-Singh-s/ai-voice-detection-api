import base64
from audio_utils import extract_features_from_base64
from model import VoiceClassifier

# load model
classifier = VoiceClassifier()
classifier.load()

with open("sample.mp3", "rb") as f:
    encoded = base64.b64encode(f.read()).decode()

features = extract_features_from_base64(encoded)
pred, conf = classifier.predict(features)

result = {
    "classification": "AI Generated" if pred == 1 else "Human",
    "confidence": round(float(conf), 3),
    "explanation": "Prediction based on MFCC audio features and trained classifier"
}

print(result)
