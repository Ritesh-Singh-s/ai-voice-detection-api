from gtts import gTTS
import os
import time

LANG_MAP = {
    "english": "en",
}

TEXT_DIR = "texts"
OUT_DIR = "data/ai"

os.makedirs(OUT_DIR, exist_ok=True)

for lang, code in LANG_MAP.items():
    text_file = os.path.join(TEXT_DIR, f"{lang}.txt")
    out_lang_dir = os.path.join(OUT_DIR, lang)
    os.makedirs(out_lang_dir, exist_ok=True)

    with open(text_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        out_path = os.path.join(out_lang_dir, f"ai_{lang}_{i:05d}.mp3")

        # 🔹 SKIP if already generated
        if os.path.exists(out_path):
            continue

        sentence = line.strip()
        if len(sentence) < 5:
            continue

        try:
            tts = gTTS(text=sentence, lang=code)
            tts.save(out_path)

            if i % 50 == 0:
                print(f"{lang}: generated {i}")

            time.sleep(0.5)  # 🔹 prevents rate limit

        except Exception as e:
            print(f"Error at {lang} line {i}: {e}")
            time.sleep(5)
            continue
