import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from detector_face_func import *
import cv2
import matplotlib.pyplot as plt


def show_img(img):
    plt.title('Result')
    plt.axis('off')
    plt.imshow(img)
    plt.show()

def detection_face(path):
    base_options = python.BaseOptions(model_asset_path='face_detector.tflite')
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)
    image = mp.Image.create_from_file(path)
    detection_result = detector.detect(image)
    image_copy = np.copy(image.numpy_view())
    annotated_image = visualize(image_copy, detection_result)
    rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    show_img(rgb_annotated_image)

path = 'FIX_LATIH_UJI/db_head_straight/fear/fear_0006.jpg'
detection_face(path)