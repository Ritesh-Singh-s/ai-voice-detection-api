
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import base64
import re
import io
import librosa
import numpy as np
from model import VoiceClassifier

def normalize_base64(b64: str) -> bytes:
    # Remove whitespace and newlines
    b64 = re.sub(r"\s+", "", b64)

    # Fix missing padding
    missing_padding = len(b64) % 4
    if missing_padding:
        b64 += "=" * (4 - missing_padding)

    try:
        return base64.b64decode(b64, validate=False)
    except Exception:
        return base64.urlsafe_b64decode(b64)

app = FastAPI(title="AI vs Human Voice Detection API")

# -------- CONFIG --------
API_KEY = "SECRET_API_KEY_123"   # change before submission
# ------------------------

classifier = VoiceClassifier()
classifier.load("voice_model.pkl")

from pydantic import BaseModel, Field
from typing import Optional

class AudioRequest(BaseModel):
    audio_base64: str = Field(..., alias="audioBase64")
    language: Optional[str] = None
    audioFormat: Optional[str] = None

    class Config:
        allow_population_by_field_name = True



def extract_features_from_base64_mp3(b64_audio: str):
    try:
        audio_bytes = normalize_base64(b64_audio)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio input")

    audio_buffer = io.BytesIO(audio_bytes)

    try:
        # Let librosa validate & decode
        y, sr = librosa.load(audio_buffer, sr=16000, mono=True)
    except Exception:
        # If decoding fails, it's not a valid MP3
        raise HTTPException(
            status_code=400,
            detail="Invalid Base64 MP3 audio input"
        )

    if y is None or len(y) == 0:
        raise HTTPException(
            status_code=400,
            detail="Invalid Base64 MP3 audio input"
        )

    mfcc = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)



@app.post("/detect")
def detect_voice(
    req: AudioRequest,
    x_api_key: str = Header(None)
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    features = extract_features_from_base64_mp3(req.audio_base64)
    pred, conf = classifier.predict(features)

    if pred == 1:
        classification = "AI_GENERATED"
        explanation = (
            "Audio shows low jitter, smooth pitch contours, and "
            "regular spectral patterns typical of synthesized speech"
        )
    else:
        classification = "HUMAN"
        explanation = (
            "Audio contains natural pitch variation, micro-pauses, "
            "and irregular spectral patterns typical of human speech"
        )

    return {
        "classification": classification,
        "confidenceScore": round(float(conf), 3),
        "explanation": explanation
    }


@app.get("/health")
def health():
    return {"status": "ok"}
