import cv2
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp
import glob

from frontalization import *

# Konfigurasi folder
source_dir = "FIX_LATIH_UJI/db_hand_rand"
before_target_dir = f'{source_dir}_before'
after_target_dir = f'{source_dir}_after'

valid_exts = (".png", ".jpg", ".jpeg", ".bmp")

image_paths = glob.glob(os.path.join(source_dir, "*", "*.*"))
print(image_paths)

for img_path in image_paths:
    if not img_path.lower().endswith(valid_exts):
        continue
    try:
        # Ambil label dan nama file
        parts = img_path.split(os.sep)
        emotion_label = parts[-2]
        filename = parts[-1]

        files_ = cv2.imread(img_path)

        flow, ori, half = half_flip(files_)
        face_only_img = cv2.cvtColor(ori, cv2.COLOR_BGR2RGB)
        half_face = cv2.cvtColor(half, cv2.COLOR_BGR2RGB)
        
        if half is not None:
            target_folder_before = os.path.join(before_target_dir, emotion_label)
            os.makedirs(target_folder_before, exist_ok=True)
            
            target_folder_after = os.path.join(after_target_dir, emotion_label)
            os.makedirs(target_folder_after, exist_ok=True)

            target_before = os.path.join(target_folder_before, filename)
            cv2.imwrite(target_before, face_only_img)     
            print(f"✔️ Proses before : {target_before}")

            target_after = os.path.join(target_folder_after, filename)
            cv2.imwrite(target_after, half_face)     
            print(f"✔️ Proses after : {target_after}")
        
        else:
            print(f"❌ Tidak ada wajah: {img_path}")

    except Exception as e:
        print(f"❌ Error proses {img_path} - {e}")
