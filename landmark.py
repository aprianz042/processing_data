import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from landmark_func import *
import cv2

# STEP 2: Create an FaceLandmarker object.
def face_landmark(path):
    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                        output_face_blendshapes=True,
                                        output_facial_transformation_matrixes=True,
                                        num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)
    image = mp.Image.create_from_file(path)
    detection_result = detector.detect(image)
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    return annotated_image, detection_result
    


#path = 'FIX_LATIH_UJI/db_head_straight/fear/fear_0006.jpg'
path = 'pemandangan.jpg'
a, b = face_landmark(path)

if b.face_landmarks:
    show_img(cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
else:
    print("tidak ada")