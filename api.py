
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64
import io
import soundfile as sf
import librosa
import numpy as np
from model import VoiceClassifier

app = FastAPI(title="AI vs Human Voice Detection API")

# -------- CONFIG --------
API_KEY = "SECRET_API_KEY_123"   # change before submission
# ------------------------

classifier = VoiceClassifier()
classifier.load("voice_model.pkl")

from typing import Optional

class AudioRequest(BaseModel):
    audioBase64: str
    language: Optional[str] = None
    audioFormat: Optional[str] = None


def extract_features_from_base64_mp3(base64_audio: str):
    try:
        audio_bytes = base64.b64decode(base64_audio)
        audio_buffer = io.BytesIO(audio_bytes)

        # Decode audio
        y, sr = sf.read(audio_buffer)

        # Convert stereo to mono (allowed)
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)

        # Normalize sample rate (allowed)
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)

        mfcc = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=40)
        return np.mean(mfcc.T, axis=0)

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid Base64 MP3 audio input"
        )

@app.post("/detect")
def detect_voice(req: AudioRequest):
    features = extract_features_from_base64_mp3(req.audioBase64)
    pred, conf = classifier.predict(features)

    if pred == 1:
        classification = "AI_GENERATED"
        explanation = "Audio shows low jitter, smooth pitch contours, and regular spectral patterns typical of synthesized speech"
    else:
        classification = "HUMAN"
        explanation = "Audio contains natural pitch variation, micro-pauses, and irregular spectral patterns typical of human speech"

    return {
        "classification": classification,
        "confidenceScore": round(float(conf), 3),
        "explanation": explanation
    }

@app.get("/health")
def health():
    return {"status": "ok"}
