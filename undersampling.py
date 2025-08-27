import os
import random
import shutil

# Folder asli dan target
source_dir = "db_faces_clean_selected"
target_dir = "db_faces_clean_selected_undersampled"
target_count = 1000  # Jumlah target paling minimal

# Format gambar valid
valid_exts = (".jpg", ".jpeg", ".png", ".bmp")

for emotion_label in os.listdir(source_dir):
    emotion_path = os.path.join(source_dir, emotion_label)
    if not os.path.isdir(emotion_path):
        continue

    # Ambil semua file gambar valid
    files = [f for f in os.listdir(emotion_path) if f.lower().endswith(valid_exts)]

    # Undersample jika lebih dari target
    if len(files) > target_count:
        files = random.sample(files, target_count)
    else:
        print(f"ℹ️ Kelas '{emotion_label}' kurang dari target ({len(files)} gambar), tidak diubah.")

    # Simpan ke folder baru
    target_emotion_path = os.path.join(target_dir, emotion_label)
    os.makedirs(target_emotion_path, exist_ok=True)

    for file in files:
        src = os.path.join(emotion_path, file)
        dst = os.path.join(target_emotion_path, file)
        shutil.copy2(src, dst)

    print(f"✔️ Kelas '{emotion_label}': {len(files)} gambar disalin ke {target_emotion_path}")
