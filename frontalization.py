############################################## Dependency #############################################################################################
from skimage.transform import PiecewiseAffineTransform, warp

from typing import Tuple, Union
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

############################################## Mediapipe Utils ########################################################################################
MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

connections = {
    "tesselation": (mp.solutions.face_mesh.FACEMESH_TESSELATION, 'cyan'),
    "contours": (mp.solutions.face_mesh.FACEMESH_CONTOURS, 'red'),
    "irises": (mp.solutions.face_mesh.FACEMESH_IRISES, 'magenta')
}

linewidths = {
    "tesselation": 0.5,  # Lebih tipis
    "contours": 2.5,  # Medium
    "irises": 1.0  # Lebih tebal
}

def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px

def visualize(image,detection_result) -> np.ndarray:
  """Draws bounding boxes and keypoints on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  annotated_image = image.copy()
  height, width, _ = image.shape

  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

    # Draw keypoints
    for keypoint in detection.keypoints:
      keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                     width, height)
      color, thickness, radius = (0, 255, 0), 2, 2
      cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    category_name = '' if category_name is None else category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return annotated_image

def draw_hand_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

def draw_face_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image

def plot_face_blendshapes_bar_graph(face_blendshapes):
  # Extract the face blendshapes category names and scores.
  face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
  face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
  # The blendshapes are ordered in decreasing score value.
  face_blendshapes_ranks = range(len(face_blendshapes_names))

  fig, ax = plt.subplots(figsize=(12, 12))
  bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
  ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
  ax.invert_yaxis()

  # Label each bar with values
  for score, patch in zip(face_blendshapes_scores, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

  ax.set_xlabel('Score')
  ax.set_title("Face Blendshapes")
  plt.tight_layout()
  plt.show()

def face_detection(image):
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    base_options = python.BaseOptions(model_asset_path='model/face_detector.tflite')
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)
    detection_result = detector.detect(img)
    image_copy = np.copy(img.numpy_view())
    annotated_image = visualize(image_copy, detection_result)
    return annotated_image, detection_result

def face_landmark(image):
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    base_options = python.BaseOptions(model_asset_path='model/face_landmarker.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options, output_face_blendshapes=True, output_facial_transformation_matrixes=True, num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)
    detection_result = detector.detect(img)
    annotated_image = draw_face_landmarks_on_image(img.numpy_view(), detection_result)
    return annotated_image, detection_result

def hand_detection(image): 
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    base_options = python.BaseOptions(model_asset_path='model/hand_model.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)
    detection_result = detector.detect(img)
    annotated_image = draw_hand_landmarks_on_image(img.numpy_view(), detection_result)
    return annotated_image, detection_result

def segmentation(image):
    BG_COLOR = (0, 0, 0) # gray
    MASK_COLOR = (255, 0, 0) # merah
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    base_options = python.BaseOptions(model_asset_path='model/selfie_multiclass.tflite')
    options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)
    segmenter = vision.ImageSegmenter.create_from_options(options)
    segmentation_result = segmenter.segment(img)
    category_mask = segmentation_result.category_mask
    image_data = img.numpy_view()
    fg_image = np.zeros(image_data.shape, dtype=np.uint8)
    fg_image[:] = MASK_COLOR
    bg_image = np.zeros(image_data.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.2
    output_image = np.where(condition, fg_image, bg_image)
    return output_image, segmentation_result

############################################## Show Image #############################################################################################
def show_img(img):
    plt.title('Result')
    plt.axis('off')
    plt.imshow(img)
    plt.show()

def show_images_grid(image_list, figsize=(12, 12)):
    num_images = len(image_list)
    max_cols = 6
    cols = min(max_cols, num_images)
    rows = math.ceil(num_images / cols)

    total_slots = rows * cols
    placeholder = np.ones((100, 100, 3), dtype=np.uint8) * 255  # putih
    while len(image_list) < total_slots:
        image_list.append((placeholder, None))
    image_list = image_list[:total_slots]

    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)
    axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]
    fig.subplots_adjust(bottom=0.05)

    for ax, (img, color) in zip(axs, image_list):
        ax.axis('off')
        if color:
            ax.imshow(cv2.cvtColor(img, color))
        else:
            ax.imshow(img)
    plt.show()

def show_images_grid_mini(image_list, figsize=(6, 6)):
    num_images = len(image_list)
    max_cols = 6
    cols = min(max_cols, num_images)
    rows = math.ceil(num_images / cols)

    total_slots = rows * cols
    placeholder = np.ones((100, 100, 3), dtype=np.uint8) * 255  # putih
    while len(image_list) < total_slots:
        image_list.append((placeholder, None))
    image_list = image_list[:total_slots]

    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)
    axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]
    fig.subplots_adjust(bottom=0.2)

    for ax, (img, color) in zip(axs, image_list):
        ax.axis('off')
        if color:
            ax.imshow(cv2.cvtColor(img, color))
        else:
            ax.imshow(img)
    plt.show()


############################################## Landmark Hand & Face ###################################################################################
def landmark_tangan(tangan):
    image_height, image_width, _ = tangan.shape
    list_hand = []
    img_copy = tangan
    hand_landmark, result_hand = hand_detection(img_copy)
    if len(result_hand.hand_landmarks) > 0:
        for hand_no, lm in enumerate(result_hand.hand_landmarks):
            list_poin = []
            for i in range(21):
                xc = int(lm[hand_no].x * image_width)
                yc = int(lm[hand_no].y * image_height)
                list_poin.append([xc, yc])
            list_hand.append(list_poin)
    return hand_landmark, list_hand

def landmark_wajah(face):
    face_lm, lm = face_landmark(face)
    if len(lm.face_landmarks):
        return face_lm, lm
    else:
        stat = 'gagal'
        return face, stat

def coordinate_landmark(wajah, landmark):
    # Load the image
    image = wajah.copy()
    poin_masking = landmark
    coordinates = np.array(poin_masking)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)    
    cv2.fillPoly(mask, [coordinates], 1)    
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image


############################################## Koreksi Head Roll ######################################################################################
def correct_roll(image):
    h, w, _ = image.shape
    img_rgb = image
    face_lm, lm = face_landmark(img_rgb)
    if not len(lm.face_landmarks):
        #print('Wajah tidak ditemukan!')
        return image
    for face_landmarks in lm.face_landmarks:
        left_eye = np.array([face_landmarks[33].x * w, face_landmarks[33].y * h])
        right_eye = np.array([face_landmarks[263].x * w, face_landmarks[263].y * h])
        dx, dy = right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1)
        corrected_image = cv2.warpAffine(image, M, (w, h))
        return corrected_image

def compute_yaw_angle(points):
    """ Hitung sudut yaw berdasarkan posisi hidung dan dagu """
    NOSE_IDX = 1  # Landmark hidung (pusat wajah)
    CHIN_IDX = 152  # Landmark dagu

    nose = points[NOSE_IDX]
    chin = points[CHIN_IDX]

    dx = chin[0] - nose[0]
    dz = chin[2] - nose[2]

    yaw_angle = np.degrees(np.arctan2(dx, dz))  # Sudut yaw dari sumbu Z
    return yaw_angle

def get_face_mesh_3d(image):
    """ Deteksi wajah dan ambil landmark 3D """
    h, w, _ = image.shape
    img_rgb = image
    
    face_lm, lm = face_landmark(img_rgb)
    if not len(lm.face_landmarks):
        print("Wajah tidak ditemukan!")
        return None, None

    face_landmarks = lm.face_landmarks[0]
    points = np.array([(l.x * w, l.y * h, l.z * w) for l in face_landmarks])
    
    return points, face_landmarks

def convert_to_2d_xy(points):
    """ Konversi dari 3D ke 2D dengan menghilangkan koordinat Z """
    points_2d_xy = points[:, :2]  # Ambil hanya X dan Y
    return points_2d_xy

def save_2d_face_ori(points_2d, task, gambar_asli):
    filename = "face_2d_result_colored.png"
    h, w = gambar_asli.shape[:2]
    
    # Buat canvas putih dengan ukuran yang sama dengan gambar asli
    canvas = np.ones_like(gambar_asli, dtype=np.uint8) * 255

    # Warna untuk masing-masing koneksi
    color_map = {
        "tesselation": (255, 0, 0),  # Biru
        "contours": (0, 0, 255),     # Merah
        "irises": (0, 255, 255)      # Kuning
    }

    # Ketebalan garis
    linewidths = {
        "tesselation": 1,
        "contours": 2,
        "irises": 1
    }

    # Pastikan koordinat integer dan dalam batas gambar
    points_int = np.round(points_2d).astype(int)
    points_int[:, 0] = np.clip(points_int[:, 0], 0, w - 1)
    points_int[:, 1] = np.clip(points_int[:, 1], 0, h - 1)

    # Gambar koneksi antar titik
    for name, (conn, _) in connections.items():
        color = color_map.get(name, (0, 0, 0))
        thickness = linewidths.get(name, 1)
        for start, end in conn:
            pt1 = tuple(points_int[start])
            pt2 = tuple(points_int[end])
            cv2.line(canvas, pt1, pt2, color, thickness)

    # Gambar titik
    for x, y in points_int:
        cv2.circle(canvas, (x, y), 2, (0, 0, 0), -1)

    if task == 'simpan':
        cv2.imwrite(filename, canvas)
        print(f"Gambar 2D disimpan sebagai {filename}")
        return None
    else:
        return canvas

def rotate(points, angle, axis='z'):
    """ Rotasi titik 3D berdasarkan sumbu (z = roll, y = yaw, x = pitch) """
    angle_rad = np.radians(angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    if axis == 'z':  # Rotasi Roll
        R = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
    elif axis == 'y':  # Rotasi Yaw
        R = np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ])
    elif axis == 'x':  # Rotasi Pitch
        R = np.array([
            [1, 0, 0],
            [0, cos_a, -sin_a],
            [0, sin_a, cos_a]
        ])
    else:
        raise ValueError("Axis harus 'z' (roll), 'y' (yaw), atau 'x' (pitch)")

    # Pusatkan titik sebelum rotasi
    center = np.mean(points, axis=0)
    rotated_points = np.dot(points - center, R.T) + center

    return rotated_points

############################################## Masking ################################################################################################
# hasil masking == ukuran asli image
def masking_img(images, f_poin, warna):
    warna_dict = {'putih': (255, 255, 255),
                  'hitam': (0, 0, 0),
                  'biru': (0, 0, 255),
                  'hijau': (0, 255, 0),
                  'merah': (255, 0, 0)}

    point_f = np.array(f_poin)
    l_face = coordinate_landmark(images, f_poin)
    l_faces = l_face.copy()
    im_bwf = cv2.fillPoly(l_faces, pts=[point_f], color=warna_dict[warna])
    return im_bwf


# hasil masking ==  ukuran wajah
def masking_img_(images, f_poin, warna):
    warna_dict = {'putih': (255, 255, 255),
                  'hitam': (0, 0, 0),
                  'biru': (0, 0, 255),
                  'hijau': (0, 255, 0),
                  'merah': (255, 0, 0)}

    point_f = np.array(f_poin)
    l_face = coordinate_landmark(images, f_poin)
    l_faces = l_face.copy()
    im_bwf = cv2.fillPoly(l_faces, pts=[point_f], color=warna_dict[warna])
    
    x, y, w, h = cv2.boundingRect(np.array(f_poin, dtype=np.int32))
    # Crop hasil sesuai bounding box
    hasil_crop = im_bwf[y:y+h, x:x+w]
    return hasil_crop

############################################## Blurring_hand ##########################################################################################
def blurring_hand(image):
    max_hands=2
    inpaint_radius=15 
    dilate_size=21
    method='NS'
    
    hand_landmark, result_hand = hand_detection(image)
    segmented, results = segmentation(image)
    h, w, _ = image.shape
    seg_binary = segmented
    mask_total = np.zeros((h,w), dtype=np.uint8)

    if len(result_hand.hand_landmarks) > 0:
        hasil = []
        for no_hand, hand_landmarks in enumerate(result_hand.hand_landmarks):
            points = []
            for lm in hand_landmarks:
                px = int(lm.x * w)
                py = int(lm.y * h)
                points.append([px, py])
            hull = cv2.convexHull(np.array(points, dtype=np.int32))
            mask_hand = np.zeros((h,w), dtype=np.uint8)
            cv2.fillPoly(mask_hand, [hull], 255)

            hand_only = cv2.bitwise_and(seg_binary, seg_binary, mask=mask_hand)
            #mask_total = cv2.bitwise_or(mask_total, hand_only)
            mask_total = hand_only

            # Dilate supaya benar-benar menutup area jari
            kernel = np.ones((dilate_size,dilate_size), np.uint8)
            mask_total = cv2.dilate(mask_total, kernel, iterations=1)
            hasil.append(mask_total)
            
        if len(hasil) > 1:
            tangan = hasil[0].copy()       
            for img in hasil[1:]:
                mask_tangan = np.any(image != [0, 0, 0], axis=-1) 
                tangan[mask_tangan] = image[mask_tangan]
        else:
            tangan = hasil[0]
    else:
        tangan = np.zeros_like(image)

    flag = cv2.INPAINT_NS if method == 'NS' else cv2.INPAINT_TELEA
    result = cv2.inpaint(image, mask_total, inpaintRadius=inpaint_radius, flags=flag)
    return result

def _blur(image):
    _, result_hand = hand_detection(image)
    h, w, _ = image.shape
    mask_hand = np.zeros((h, w), dtype=np.uint8)

    if len(result_hand.hand_landmarks) > 0:
        for no_hand, hand_landmarks in enumerate(result_hand.hand_landmarks):
            points = []
            for lm in hand_landmarks:
                px = int(lm.x * w)
                py = int(lm.y * h)
                points.append([px, py])
            hull = cv2.convexHull(np.array(points, dtype=np.int32))
            cv2.fillPoly(mask_hand, [hull], 255)
            kernel = np.ones((30, 30), np.uint8)
            mask_hand = cv2.dilate(mask_hand, kernel, iterations=1)

    blurred_image = cv2.GaussianBlur(image, (151, 151), 0)
    blurred_area = cv2.bitwise_and(blurred_image, blurred_image, mask=mask_hand)
    background = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask_hand))
    final_result = cv2.add(background, blurred_area)

    return final_result

def neighborhood_inpainting(image):
    _, result_hand = hand_detection(image)
    h, w, _ = image.shape

    if len(result_hand.hand_landmarks) > 0:
        hasil = []
        for _, hand_landmarks in enumerate(result_hand.hand_landmarks):
            points = []
            for lm in hand_landmarks:
                px = int(lm.x * w)
                py = int(lm.y * h)
                points.append([px, py])
            hull = cv2.convexHull(np.array(points, dtype=np.int32))
            
            mask_temp = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask_temp, [hull], 255)

            kernel = np.ones((30, 30), np.uint8)
            mask_temp = cv2.dilate(mask_temp, kernel, iterations=1)

            inpainted_image = cv2.inpaint(image, mask_temp, inpaintRadius=10, flags=cv2.INPAINT_TELEA)
            hasil.append(inpainted_image)

        if len(hasil) > 1:
            wajah = hasil[0]
            for img in hasil[1:]:
                wajah = cv2.addWeighted(wajah, 0.5, img, 0.5, 0)
                 
        else:
            wajah = hasil[0]
    else:
        wajah = image
    
    return wajah


def masking_tangan_canvas_hitam(img, dilate_size=21):
    hand_landmark, result_hand = hand_detection(img)
    segmented, results = segmentation(img)
    h, w, _ = img.shape
    seg_binary = segmented
    mask_total = np.zeros((h,w), dtype=np.uint8)

    if len(result_hand.hand_landmarks) > 0:
        hasil = []
        for no_hand, hand_landmarks in enumerate(result_hand.hand_landmarks):
            points = []
            for lm in hand_landmarks:
                px = int(lm.x * w)
                py = int(lm.y * h)
                points.append([px, py])
            hull = cv2.convexHull(np.array(points, dtype=np.int32))
            mask_hand = np.zeros((h,w), dtype=np.uint8)
            cv2.fillPoly(mask_hand, [hull], 255)
    
            hand_only = cv2.bitwise_and(seg_binary, seg_binary, mask=mask_hand)
            #mask_total = cv2.bitwise_or(mask_total, hand_only)
            mask_total = hand_only
    
            # Pertebal mask agar menutup semua jari
            kernel = np.ones((dilate_size,dilate_size), np.uint8)
            mask_total = cv2.dilate(mask_total, kernel, iterations=1)
            hasil.append(mask_total)
            
        if len(hasil) > 1:
            tangan = hasil[0].copy()       
            for img in hasil[1:]:
                mask_tangan = np.any(img != [0, 0, 0], axis=-1) 
                tangan[mask_tangan] = img[mask_tangan]
        else:
            tangan = hasil[0]
    else:
        tangan = np.zeros_like(img)
        
    return tangan

############################################## Crop Wajah #############################################################################################
def potong_area(img, landmark_region):
    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.array(landmark_region, dtype=np.int32), (255, 255, 255))
    hasil = cv2.bitwise_and(img, mask)
    return hasil

def potong_area_(img, landmark_region):
    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.array(landmark_region, dtype=np.int32), (255, 255, 255))
    # Ambil area wajah saja
    hasil = cv2.bitwise_and(img, mask)
    # Hitung bounding box dari polygon landmark
    x, y, w, h = cv2.boundingRect(np.array(landmark_region, dtype=np.int32))
    # Crop hasil sesuai bounding box
    hasil_crop = hasil[y:y+h, x:x+w]
    return hasil_crop

############################################## Luas Wajah #############################################################################################
def luas_wajah(img):
    mask_putih = np.all(img == [255, 255, 255], axis=-1)
    luas_putih = np.sum(mask_putih)
    return luas_putih

############################################## Potong Tengah ##########################################################################################
def spatial_divide(img, xh):
    img_height, img_width, _ = img.shape
    cropped_image_l = img[0:img_height, 0:xh]
    cropped_image_r = img[0:img_height, xh:img_width]
    return cropped_image_l, cropped_image_r

############################################## Face Detection #########################################################################################
def crop_faces_from_image(img_input): #haarcascade
    # Load Haar cascade
    face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
    if not isinstance(img_input, np.ndarray):
        raise ValueError("Input harus berupa image array (numpy.ndarray)")
    #gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_input, scaleFactor=1.2, minNeighbors=6, minSize=(40, 40))
    if len(faces) == 0:
        print("Tidak ada wajah terdeteksi.")
        return []
    cropped_faces = []
    for (x, y, w, h) in faces:
        face = img_input[y:y+h, x:x+w]
        cropped_faces.append(face)
    return cropped_faces


############################################## Combine Hands ##########################################################################################
def combine_hand(hand1, hand2):
    mask_tangan = np.any(hand1 != [0, 0, 0], axis=-1)
    hasil = hand2.copy()
    hasil[mask_tangan] = hand1[mask_tangan]
    return hasil

############################################## Poin Landmark Wajah ####################################################################################
def list_poin_tangan(list_poin):
    poin_hand = [list_poin[0], list_poin[1], list_poin[2], list_poin[3], 
                    list_poin[4], list_poin[5], list_poin[6], list_poin[7], 
                    list_poin[8], list_poin[12], list_poin[16], 
                    list_poin[20], list_poin[19], list_poin[18],list_poin[17]]
    return poin_hand

def list_poin_wajah(dua_D):
    face_poin = [dua_D[152], dua_D[148], dua_D[176], dua_D[149], dua_D[150], dua_D[136], dua_D[172], dua_D[58], dua_D[132],
                  dua_D[93], dua_D[234], dua_D[127], dua_D[162], dua_D[21], dua_D[54], dua_D[103], dua_D[67], dua_D[109],
                  dua_D[10], dua_D[338], dua_D[297], dua_D[332], dua_D[284], dua_D[251], dua_D[389], dua_D[356], dua_D[454],
                  dua_D[323], dua_D[361], dua_D[288], dua_D[397], dua_D[365], dua_D[379], dua_D[378], dua_D[400], dua_D[377]
                 ]
    return face_poin

def wajah_kiri(dua_D):
    face_poin = [dua_D[152], dua_D[175], dua_D[199], dua_D[200], dua_D[18], dua_D[17], dua_D[16], dua_D[15], dua_D[14], dua_D[13], dua_D[12], 
                 dua_D[11], dua_D[0], dua_D[164], dua_D[2], dua_D[94], dua_D[4], dua_D[5], dua_D[195], dua_D[197], dua_D[6], dua_D[168], 
                 dua_D[8], dua_D[9], dua_D[151], dua_D[10], 
                 dua_D[109], dua_D[67], dua_D[103], dua_D[54], dua_D[21], dua_D[162], dua_D[127], dua_D[234], dua_D[93], dua_D[132], dua_D[58], 
                 dua_D[172], dua_D[136], dua_D[150], dua_D[149], dua_D[176], dua_D[148]
                 ]
    return face_poin

def wajah_kanan(dua_D):
    face_poin = [dua_D[152], dua_D[175], dua_D[199], dua_D[200], dua_D[18], dua_D[17], dua_D[16], dua_D[15], dua_D[14], dua_D[13], dua_D[12], 
                 dua_D[11], dua_D[0], dua_D[164], dua_D[2], dua_D[94], dua_D[4], dua_D[5], dua_D[195], dua_D[197], dua_D[6], dua_D[168], 
                 dua_D[8], dua_D[9], dua_D[151], dua_D[10], 
                 dua_D[338], dua_D[297], dua_D[332], dua_D[284], dua_D[251], dua_D[389], dua_D[356], dua_D[454], dua_D[323], dua_D[361], 
                 dua_D[288], dua_D[397], dua_D[365], dua_D[379], dua_D[378], dua_D[400], dua_D[377] 
                 ]
    return face_poin

def warp_poin(point): #poin warping baru (x lurus)
    poin = point.copy()
    xp, yp = poin[13]
    poin[:26] = [(xp, y) for (_, y) in poin[:26]]
    return poin

############################################## Warping Wajah ke Warp Poin #############################################################################
def images_warping(data_wajah, data_landmark_src, data_landmark_dst):
    im = data_wajah
    #im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_rgb = im
    
    src_landmarks = data_landmark_src  # bentuk (N, 2)
    dst_landmarks = data_landmark_dst  # bentuk (N, 2)
    
    image_shape = im_rgb.shape
    height, width = image_shape[:2]
    
    tform = PiecewiseAffineTransform()
    tform.estimate(dst_landmarks, src_landmarks)  # dari target ke source
    
    warped = warp(im_rgb, tform, output_shape=(height, width))
    warped = (warped * 255).astype(np.uint8)
    return warped

############################################## enhance_contrast #######################################################################################
def enhance_contrast(frame):
    #lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lab = frame
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

############################################## Main Function ##########################################################################################
def half_flip(img):  
    images = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = images.shape 
    output_list = []                                          # variabel list OUTPUT

    #images = enhance_contrast(images)
    
    if img_h < 500:
        new_height = 500                                          # ukuran tinggi image (sesuaikan)
        (h, w) = images.shape[:2]
        aspect_ratio = w / h
        new_width = int(new_height * aspect_ratio)
        images = cv2.resize(images, (new_width, new_height))      # resize tinggi image ke ukuran baru

    img_ori = images.copy()                                   # img_ori = gambar asli    
    img_roll = correct_roll(img_ori)                          # img_roll = headpose yang sudah di luruskan (yang diproses selanjutnya)
    img_r = img_roll.copy()                                   # img_roll yang tidak di proses (untuk visualisasi)

    img_rr = img_roll.copy()
    img_rr = neighborhood_inpainting(img_rr)
    
    #output_list.append((img_rr, None))
    
    points_3d, face_landmarkss = get_face_mesh_3d(img_rr)
    if points_3d is not None and face_landmarkss is not None:
        yaw_angle = compute_yaw_angle(points_3d)
        points_3d = rotate(points_3d, -yaw_angle, axis='y')  
        points_2d_xy = convert_to_2d_xy(points_3d)
        yaw = save_2d_face_ori(points_2d_xy, 'ubah', img_rr)
        #output_list.append((yaw, None))

    output_list.append((img_ori, None))          # ----> img_ori tanpa landmark
    #output_list.append((img_r, cv2.COLOR_RGB2BGR))           # ----> img_roll tanpa landmark
    output_list.append((img_rr, None))
    
    img_height, img_width, _ = img_roll.shape 
    
    ######################## Proses Landmarking Tangan - START ########################################################################################
    mesh_hand, list_hand = landmark_tangan(img_roll)          # proses landmarking tangan
    if list_hand is not None:                                    # jika tangan terdeteksi hanya 1
        hand_masked = masking_tangan_canvas_hitam(img_roll)
        hand_masked_ = hand_masked
        output_list.append((hand_masked, None))              # ----> masking 1 tangan
        combine_face_hand = 'True'
    else:                                                     # jika tidak ada tangan terdeteksi, maka tidak ada proses landmarking tangan
        combine_face_hand = False                             # tidak ada proses penggabungan dengan wajah 
        print('tangan tidak terdeteksi')
    ######################## END - Proses Landmarking Tangan ##########################################################################################

    ######################## Proses Wajah - START #####################################################################################################
    mesh_wajah, landmark_face = landmark_wajah(img_roll)      # proses landmarking wajah dari gambar yang sudah corect roll (img_roll)
    if len(landmark_face.face_landmarks[0]) == 0:                               # jika landmark == 0 atau tidak ada wajah terdeteksi maka proses selesai
        return None
    else:
        dua_D = []                                            # proses mengkonversi koordinat landmark (x,y,z) dari 3D menjadi 2D
        for i, lm in enumerate(landmark_face.face_landmarks[0]):
            ix = int(lm.x * img_width)
            iy = int(lm.y * img_height)
            dua_D.append([ix, iy])                            # dud_D = koordinat wajah 2D
        
        ######################## landmarking full wajah - START #######################################################################################
        poin_wajah_full = list_poin_wajah(dua_D)
        wajah_full_masked = masking_img(img_roll, poin_wajah_full, 'putih')
        #landmark_warp_full = warp_poin(poin_wajah_full)                # proses warping poin objek mask putih tengah wajah
        wajah_full_masked_ = masking_img_(img_roll, poin_wajah_full, 'putih')
        warpp_full = images_warping(img_r, poin_wajah_full, poin_wajah_full)
        face_ori = potong_area_(warpp_full, poin_wajah_full)
        face_ori = cv2.resize(face_ori, (128, 128))
        output_list.append((warpp_full, None))
        #output_list.append((face_ori, None))   # ----> hasil warping wajah asli full
        ######################## END - landmarking full wajah #########################################################################################

        ######################## landmarking wajah kiri - START #######################################################################################
        poin_wajah_kiri = wajah_kiri(dua_D)
        face_kiri_masked = masking_img(img_roll, poin_wajah_kiri, 'putih')
        landmark_warp_kiri = warp_poin(poin_wajah_kiri)                # proses warping poin objek mask putih tengah wajah
        face_kiri_masked_ = masking_img_(img_roll, landmark_warp_kiri, 'putih')
        #output_list.append((face_kiri_masked, cv2.COLOR_RGB2BGR))      # masking wajah dengan ukuran seluruh img
        #output_list.append((face_kiri_masked_, cv2.COLOR_RGB2BGR))    # masking wajah dengan hanya ukuran wajah 
        ######################## END - landmarking wajah kiri #########################################################################################
        
        
        ######################## landmarking wajah kanan - START ######################################################################################
        poin_wajah_kanan = wajah_kanan(dua_D)
        face_kanan_masked = masking_img(img_roll, poin_wajah_kanan, 'putih')
        landmark_warp_kanan = warp_poin(poin_wajah_kanan)              # proses warping poin objek mask putih tengah wajah
        face_kanan_masked_ = masking_img_(img_roll, landmark_warp_kanan, 'putih')
        #output_list.append((face_kanan_masked, cv2.COLOR_RGB2BGR))     # masking wajah dengan ukuran seluruh img
        #output_list.append((face_kanan_masked_, cv2.COLOR_RGB2BGR))   # masking wajah dengan hanya ukuran wajah 
        ######################## END - landmarking wajah kanan ########################################################################################


        ######################## Combine Wajah & Hand - START #########################################################################################
        if combine_face_hand == 'True':       
            target_shape = (hand_masked.shape[1], hand_masked.shape[0])   

            if wajah_full_masked.shape != hand_masked_[0].shape:
                wajah_full_masked = cv2.resize(wajah_full_masked, target_shape)
            mask_hand_full = np.any(hand_masked != [0, 0, 0], axis=-1)
            hasil_full = wajah_full_masked.copy()
            hasil_full[mask_hand_full] = hand_masked[mask_hand_full]
            
            if face_kanan_masked.shape != hand_masked.shape:
                face_kanan_masked = cv2.resize(face_kanan_masked, target_shape)
            # Mask area non-hitam dari hand_masked
            mask_hand_kanan = np.any(hand_masked != [0, 0, 0], axis=-1)
            hasil_kanan = face_kanan_masked.copy()
            hasil_kanan[mask_hand_kanan] = hand_masked[mask_hand_kanan]

            if face_kiri_masked.shape != hand_masked.shape:
                face_kiri_masked = cv2.resize(face_kiri_masked, target_shape)
                
            mask_hand_kiri = np.any(hand_masked != [0, 0, 0], axis=-1)    # Mask area non-hitam dari hand_masked
            
            hasil_kiri = face_kiri_masked.copy()                          # hasil_kanan = tangan kanan dan wajah kanan
            hasil_kiri[mask_hand_kiri] = hand_masked[mask_hand_kiri]      # hasil_kiri = tangan kiri dan wajah kanan
        
        # jika tidak ada objek tangan langsung pakai gambar marking wajah kiri kanan
        else:                              
            hasil_kanan = face_kanan_masked.copy()                        # hasil_kiri = tangan kiri dan wajah kanan               
            hasil_kiri = face_kiri_masked.copy()                          # hasil_kiri = tangan kiri dan wajah kanan                

        output_list.append((hasil_full, None))
        output_list.append((hasil_kiri, None))
        output_list.append((hasil_kanan, None))
        ######################## END - Combine Wajah & Hand ###########################################################################################
        
        ######################## warping wajah asli kanan - START #####################################################################################
        poin_wajah_kanan = wajah_kanan(dua_D)                            
        landmark_warp_kanan = warp_poin(poin_wajah_kanan)
        warpp_kanan = images_warping(img_rr, poin_wajah_kanan, landmark_warp_kanan)
        output_list.append((warpp_kanan, None))                      # ----> hasil warping wajah asli kanan
        ######################## END - warping wajah asli kanan #######################################################################################
        
        ######################## warping wajah asli kiri - START ######################################################################################
        poin_wajah_kiri = wajah_kiri(dua_D)
        landmark_warp_kiri = warp_poin(poin_wajah_kiri)
        warpp_kiri = images_warping(img_rr, poin_wajah_kiri, landmark_warp_kiri)
        output_list.append((warpp_kiri, None))                       # ----> hasil warping wajah asli kiri
        ######################## END - warping wajah asli kiri ########################################################################################

        
        ######################## proses membandingkan luas wajah kanan dan kiri  - START ##############################################################
        luas_kiri = luas_wajah(hasil_kiri)
        luas_kanan = luas_wajah(hasil_kanan)
        if luas_kiri > luas_kanan:                                        # jika lebih luas kiri maka bagian kiri wajah yang dipakai
            half_face = warpp_kiri
            half_face = potong_area_(half_face, landmark_warp_kiri)       # proses memotong hanya bagian wajah
            flip_image = cv2.flip(half_face, 1)                           # flip wajah
            combine = np.concatenate((half_face, flip_image), axis=1)     # gabungkan wajah kanan kiri
            output_list.append((warpp_kiri, None))                        # ----> output wajah kiri asli yang sudah diwarping
            
        else:                                                             # jika lebih luas kanan maka bagian kanan wajah yang dipakai
            half_face = warpp_kanan
            half_face = potong_area_(half_face, landmark_warp_kanan)      # proses memotong hanya bagian wajah
            flip_image = cv2.flip(half_face, 1)                           # flip wajah
            combine = np.concatenate((flip_image, half_face), axis=1)     # gabungkan wajah kanan kanan
            output_list.append((warpp_kanan, None))                       # ----> output wajah kanan asli yang sudah diwarping
        ######################## END - proses membandingkan luas wajah kanan dan kiri #################################################################

        ######################## proses resize & save - START #########################################################################################
        flip_output = combine                                             # flip_output = output proses wajah
        flip_output = cv2.resize(flip_output, (128, 128))                 # resize gambar wajah output
        #cv2.imwrite("hasil_crop.jpg", flip_output)                       # jika outputnya ingin disimpan
        output_list.append((flip_output, None))                           # ----> output wajah yang sudah diresize
        ######################## END - proses resize & save ##########################################################################################
   
    #output_ = output_list
    return output_list, face_ori, flip_output
    ######################## END - Proses Wajah ######################################################################################################

       