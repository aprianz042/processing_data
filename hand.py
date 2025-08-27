import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
Image = mp.Image
ImageFormat = mp.ImageFormat

def tangan(image): 
    model_path = 'hand_model.task'

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE)
        
    with HandLandmarker.create_from_options(options) as landmarker:           
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = Image(ImageFormat.SRGB, image_rgb)
        results = landmarker.detect(mp_image)
        if results.hand_landmarks:
            for landmarks in results.hand_landmarks:
                for landmark in landmarks:
                    x = int(landmark.x * image.shape[1])
                    y = int(landmark.y * image.shape[0])
                    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Gambar titik dengan warna hijau
        return image

def main_():    
    image = cv2.imread('FIX_LATIH_UJI/db_hand_rand/angry/angry_0955_hand.jpg')
    output = tangan(image)
    return output


gam = main_()
cv2.imshow('Hand Landmarks', gam)
cv2.waitKey(0)
cv2.destroyAllWindows()

