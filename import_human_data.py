import os
import shutil

SOURCE = r"C:\Users\Ritesh Singh\Downloads\human_dataset\english"
TARGET = "data/human/english"

os.makedirs(TARGET, exist_ok=True)

print("Reading from:", SOURCE)

for file in os.listdir(SOURCE):
    if file.lower().endswith(".wav"):
        src_file = os.path.join(SOURCE, file)
        dst_file = os.path.join(TARGET, file)

        # Skip if already exists
        if os.path.exists(dst_file):
            print(f"Skipped (already exists): {file}")
            continue

        shutil.copy(src_file, dst_file)
        print(f"Copied: {file}")

print("\n✅ All English WAV files imported (existing files skipped).")
