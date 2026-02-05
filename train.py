import os
import numpy as np
from audio_utils import extract_features
from model import VoiceClassifier

DATASET_PATH = "data"

X = []
y = []

good = 0
bad = 0

for label, category in enumerate(["human", "ai"]):
    category_path = os.path.join(DATASET_PATH, category)
    print(f"\nProcessing category: {category}")

    for lang in os.listdir(category_path):
        lang_path = os.path.join(category_path, lang)
        print(f"  Language: {lang}")

        for file in os.listdir(lang_path):
            if not file.lower().endswith((".mp3", ".wav")):
                continue

            file_path = os.path.join(lang_path, file)

            features = extract_features(file_path)

            if features is None:
                bad += 1
                continue

            X.append(features)
            y.append(label)
            good += 1

            if good % 50 == 0:
                print(f"    Processed {good} valid files...")

print("\nSummary:")
print("Valid samples:", good)
print("Skipped samples:", bad)

X = np.array(X)
y = np.array(y)

classifier = VoiceClassifier()
classifier.train(X, y)
classifier.save()

print("\n✅ Model trained and saved successfully")
