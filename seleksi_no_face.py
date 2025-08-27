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

image_paths = glob.glob(os.path.join(after_target_dir, "*", "*.*"))
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

        face_lm, lm = face_landmark(files_)
        if not len(lm.face_landmarks):
            target_folder_before = os.path.join(before_target_dir, emotion_label)
            os.makedirs(target_folder_before, exist_ok=True)
            
            target_folder_after = os.path.join(after_target_dir, emotion_label)
            os.makedirs(target_folder_after, exist_ok=True)

            target_before = os.path.join(target_folder_before, filename)
            os.remove(target_before) 
            print(f"✔️ Delete before : {target_before}")

            target_after = os.path.join(target_folder_after, filename)
            os.remove(target_after)  
            print(f"✔️ Delete after : {target_after}")
        
        else:
            print(f"❌ Pass: {img_path}")

    except Exception as e:
        print(f"❌ Error proses {img_path} - {e}")
