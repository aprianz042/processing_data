############################################## Dependency #############################################################################################
import cv2
import numpy as np
import mediapipe as mp
from skimage.transform import PiecewiseAffineTransform, warp

############################################## Mediapipe Utils ########################################################################################
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_detection = mp.solutions.face_detection
mp_selfie_segmentation = mp.solutions.selfie_segmentation

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.2)

mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)

selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Ambil daftar koneksi untuk Tesselation, Contours, dan Irises
connections = {
    "tesselation": (mp_face_mesh.FACEMESH_TESSELATION, 'cyan'),
    "contours": (mp_face_mesh.FACEMESH_CONTOURS, 'red'),
    "irises": (mp_face_mesh.FACEMESH_IRISES, 'magenta')
}

# Atur linewidth untuk setiap koneksi
linewidths = {
    "tesselation": 0.5,  # Lebih tipis
    "contours": 2.5,  # Medium
    "irises": 1.0  # Lebih tebal
}

############################################## Landmark Hand & Face ###################################################################################
def landmark_tangan(tangan):
    image_height, image_width, _ = tangan.shape
    list_hand = []
    
    img_copy = tangan.copy()
    results = hands.process(cv2.cvtColor(tangan, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
            list_poin = []
            mp_drawing.draw_landmarks(image = img_copy, landmark_list = hand_landmarks, connections = mp_hands.HAND_CONNECTIONS)
            for i in range(21):
                xc = int(hand_landmarks.landmark[mp_hands.HandLandmark(i).value].x * image_width)
                yc = int(hand_landmarks.landmark[mp_hands.HandLandmark(i).value].y * image_height)
                list_poin.append([xc, yc])
            list_hand.append(list_poin)
    return img_copy[:,:,::-1], list_hand

def landmark_wajah(face):
    annotated_image = face
    results = face_mesh.process(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        stat = None
        return annotated_image, stat
    else:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(image=annotated_image,
                                      landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_TESSELATION,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(image=annotated_image,
                                      landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_CONTOURS,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(image=annotated_image,
                                      landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_IRISES,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
        return annotated_image, results.multi_face_landmarks[0].landmark

def coordinate_landmark(wajah, landmark):
    image = wajah.copy()
    poin_masking = landmark
    coordinates = np.array(poin_masking)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)    
    cv2.fillPoly(mask, [coordinates], 1)    
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image

############################################## Koreksi Head Roll ######################################################################################
def correct_roll(image):
    image = image.copy()
    h, w, _ = image.shape
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    if not results.multi_face_landmarks:
        print('Wajah tidak ditemukan!')
        detected = False
        return image, detected
    for face_landmarks in results.multi_face_landmarks:
        left_eye = np.array([face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h])
        right_eye = np.array([face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h])
        dx, dy = right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1)
        corrected_image = cv2.warpAffine(image, M, (w, h))
        detected = True
        return corrected_image, detected

def compute_yaw_angle(points):
    NOSE_IDX = 1  # Landmark hidung (pusat wajah)
    CHIN_IDX = 152  # Landmark dagu
    nose = points[NOSE_IDX]
    chin = points[CHIN_IDX]
    dx = chin[0] - nose[0]
    dz = chin[2] - nose[2]
    yaw_angle = np.degrees(np.arctan2(dx, dz))
    return yaw_angle

def convert_to_2d_xy(points):
    points_2d_xy = points[:, :2]  # Ambil hanya X dan Y
    return points_2d_xy

def save_2d_face_ori(points_2d, task, gambar_asli):
    filename = "face_2d_result_colored.png"
    h, w = gambar_asli.shape[:2]    
    canvas = np.ones_like(gambar_asli, dtype=np.uint8) * 255

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
    inpaint_radius=15 
    dilate_size=31
    method='NS'
    
    h, w, _ = image.shape
    
    results_seg = selfie_segmentation.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    seg_mask = results_seg.segmentation_mask
    seg_binary = (seg_mask > 0.5).astype(np.uint8) * 255

    results_hand = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    mask_total = np.zeros((h,w), dtype=np.uint8)

    if results_hand.multi_hand_landmarks:
        for hand_landmarks in results_hand.multi_hand_landmarks:
            points = []
            indices_to_process = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 19, 18, 17]
            
            for idx in indices_to_process:
                lm = hand_landmarks.landmark[idx] 
                px = int(lm.x * w)  
                py = int(lm.y * h)
                points.append([px, py])
            
            mask_hand = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask_hand, [np.array(points, dtype=np.int32)], 255) 

            hand_only = cv2.bitwise_and(seg_binary, seg_binary, mask=mask_hand)
            mask_total = cv2.bitwise_or(mask_total, hand_only)
    else:
        return image

    kernel = np.ones((dilate_size,dilate_size), np.uint8)
    mask_total = cv2.dilate(mask_total, kernel, iterations=1)

    flag = cv2.INPAINT_NS if method == 'NS' else cv2.INPAINT_TELEA
    result = cv2.inpaint(image, mask_total, inpaintRadius=inpaint_radius, flags=flag)
    return result

def masking_tangan_canvas_hitam(image):
    warna='merah'
    max_hands=2
    dilate_size=21
    
    warna_dict = {'putih': (255, 255, 255),
                  'hitam': (0, 0, 0),
                  'biru': (0, 0, 255),
                  'hijau': (0, 255, 0),
                  'merah': (255, 0, 0)}
    
    h, w, _ = image.shape

    results_seg = selfie_segmentation.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    seg_mask = results_seg.segmentation_mask
    seg_binary = (seg_mask > 0.5).astype(np.uint8) * 255

    results_hand = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    mask_total = np.zeros((h,w), dtype=np.uint8)

    if results_hand.multi_hand_landmarks:
        for hand_landmarks in results_hand.multi_hand_landmarks:
            points = []
            indices_to_process = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 19, 18, 17]
            
            for idx in indices_to_process:
                lm = hand_landmarks.landmark[idx]  # Ambil landmark berdasarkan indeks
                px = int(lm.x * w)  # Konversi ke koordinat piksel
                py = int(lm.y * h)
                points.append([px, py])
            
            mask_hand = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask_hand, [np.array(points, dtype=np.int32)], 255) 

            hand_only = cv2.bitwise_and(seg_binary, seg_binary, mask=mask_hand)
            mask_total = cv2.bitwise_or(mask_total, hand_only)
    else:
        return np.zeros_like(image)

    kernel = np.ones((dilate_size,dilate_size), np.uint8)
    mask_total = cv2.dilate(mask_total, kernel, iterations=1)

    canvas = np.zeros_like(image)
    canvas[mask_total == 255] = warna_dict[warna]

    return canvas

############################################## Crop Wajah #############################################################################################
def potong_area(img, landmark_region):
    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.array(landmark_region, dtype=np.int32), (255, 255, 255))
    hasil = cv2.bitwise_and(img, mask)
    return hasil

def potong_area_(img, landmark_region):
    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.array(landmark_region, dtype=np.int32), (255, 255, 255))
    hasil = cv2.bitwise_and(img, mask)
    x, y, w, h = cv2.boundingRect(np.array(landmark_region, dtype=np.int32))
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
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    src_landmarks = data_landmark_src  # bentuk (N, 2)
    dst_landmarks = data_landmark_dst  # bentuk (N, 2)
    
    image_shape = im_rgb.shape
    height, width = image_shape[:2]

    tform = PiecewiseAffineTransform()
    tform.estimate(dst_landmarks, src_landmarks)  # dari target ke source
    
    warped = warp(im_rgb, tform, output_shape=(height, width))
    warped = (warped * 255).astype(np.uint8)
    return warped
    
############################################## Main Function ##########################################################################################
def half_flip(img):  
    images = cv2.imread(img) 
    images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)     
    img_h, img_w, _ = images.shape 
    
    if img_h < 500:
        new_height = 500                                          # ukuran tinggi image (sesuaikan)
        (h, w) = images.shape[:2]
        aspect_ratio = w / h
        new_width = int(new_height * aspect_ratio)
        images = cv2.resize(images, (new_width, new_height))      # resize tinggi image ke ukuran baru
    
    img_roll, face_detected = correct_roll(images)                          # img_roll = headpose yang sudah di luruskan (yang diproses selanjutnya)
    if face_detected == True:
        img_r = img_roll.copy()                                  # img_roll yang tidak di proses (untuk visualisasi)
        img_rr = img_roll.copy()
        img_rr = blurring_hand(img_rr)
        
        img_height, img_width, _ = img_roll.shape 
        
        ######################## Proses Landmarking Tangan - START ########################################################################################
        _, list_hand = landmark_tangan(img_roll)          # proses landmarking tangan
        sum_of_hand = len(list_hand)
        if sum_of_hand > 0:                                    # jika tangan terdeteksi hanya 1
            hand_masked = masking_tangan_canvas_hitam(img_roll)
            hand_masked_ = hand_masked
            combine_face_hand = 'True'
        else:                                                     # jika tidak ada tangan terdeteksi, maka tidak ada proses landmarking tangan
            combine_face_hand = False                             # tidak ada proses penggabungan dengan wajah 
        ######################## END - Proses Landmarking Tangan ##########################################################################################

        ######################## Proses Wajah - START #####################################################################################################
        _, landmark_face = landmark_wajah(img_roll)      # proses landmarking wajah dari gambar yang sudah corect roll (img_roll)
        #if len(landmark_face) == 0:                               # jika landmark == 0 atau tidak ada wajah terdeteksi maka proses selesai
        #    return images
        #else:
        if landmark_face:
            dua_D = []                                            # proses mengkonversi koordinat landmark (x,y,z) dari 3D menjadi 2D
            for i in range(len(landmark_face)):
                ix = int(landmark_face[i].x * img_width)
                iy = int(landmark_face[i].y * img_height)
                dua_D.append([ix, iy])                            # dud_D = koordinat wajah 2D

            ######################## landmarking full wajah - START #######################################################################################
            poin_wajah_full = list_poin_wajah(dua_D)
            wajah_full_masked = masking_img(img_roll, poin_wajah_full, 'putih')
            #warpp_full = images_warping(img_r, poin_wajah_full, poin_wajah_full)
            #face_ori = potong_area_(warpp_full, poin_wajah_full)
            #face_ori = cv2.resize(face_ori, (224, 224))
            ######################## END - landmarking full wajah #########################################################################################
                
            
            ######################## landmarking wajah kiri - START #######################################################################################
            poin_wajah_kiri = wajah_kiri(dua_D)
            face_kiri_masked = masking_img(img_roll, poin_wajah_kiri, 'putih')
            landmark_warp_kiri = warp_poin(poin_wajah_kiri)                # proses warping poin objek mask putih tengah wajah
            ######################## END - landmarking wajah kiri #########################################################################################
            
            
            ######################## landmarking wajah kanan - START ######################################################################################
            poin_wajah_kanan = wajah_kanan(dua_D)
            face_kanan_masked = masking_img(img_roll, poin_wajah_kanan, 'putih')
            landmark_warp_kanan = warp_poin(poin_wajah_kanan)              # proses warping poin objek mask putih tengah wajah
            ######################## END - landmarking wajah kanan ########################################################################################


            ######################## Combine Wajah & Hand - START #########################################################################################
            # jika ada objek tangan, maka digabung dengan wajah
            if combine_face_hand == 'True':       
                #target_shape = (hand_masked.shape[1], hand_masked.shape[0])   
                #if wajah_full_masked.shape != hand_masked_[0].shape:
                #    wajah_full_masked = cv2.resize(wajah_full_masked, target_shape)
                
                mask_hand_full = np.any(hand_masked != [0, 0, 0], axis=-1)
                hasil_full = wajah_full_masked
                hasil_full[mask_hand_full] = hand_masked[mask_hand_full]
                
                mask_hand_kanan = np.any(hand_masked != [0, 0, 0], axis=-1)
                hasil_kanan = face_kanan_masked
                hasil_kanan[mask_hand_kanan] = hand_masked[mask_hand_kanan]
        
                mask_hand_kiri = np.any(hand_masked != [0, 0, 0], axis=-1)    # Mask area non-hitam dari hand_masked
                
                hasil_kiri = face_kiri_masked                         # hasil_kanan = tangan kanan dan wajah kanan
                hasil_kiri[mask_hand_kiri] = hand_masked[mask_hand_kiri]      # hasil_kiri = tangan kiri dan wajah kanan
            
            # jika tidak ada objek tangan langsung pakai gambar marking wajah kiri kanan
            else:                              
                hasil_kanan = face_kanan_masked                        # hasil_kiri = tangan kiri dan wajah kanan               
                hasil_kiri = face_kiri_masked                          # hasil_kiri = tangan kiri dan wajah kanan                
            ######################## END - Combine Wajah & Hand ###########################################################################################
            
            ######################## warping wajah asli kanan - START #####################################################################################                            
            landmark_warp_kanan = warp_poin(poin_wajah_kanan)
            warpp_kanan = images_warping(img_rr, poin_wajah_kanan, landmark_warp_kanan)
            ######################## END - warping wajah asli kanan #######################################################################################
            
            ######################## warping wajah asli kiri - START ######################################################################################
            landmark_warp_kiri = warp_poin(poin_wajah_kiri)
            warpp_kiri = images_warping(img_rr, poin_wajah_kiri, landmark_warp_kiri)
            ######################## END - warping wajah asli kiri ########################################################################################
            
            ######################## proses membandingkan luas wajah kanan dan kiri  - START ##############################################################
            luas_kiri = luas_wajah(hasil_kiri)
            luas_kanan = luas_wajah(hasil_kanan)
            if luas_kiri > luas_kanan:                                        # jika lebih luas kiri maka bagian kiri wajah yang dipakai
                half_face = warpp_kiri
                half_face = potong_area_(half_face, landmark_warp_kiri)       # proses memotong hanya bagian wajah
                flip_image = cv2.flip(half_face, 1)                           # flip wajah
                combine = np.concatenate((half_face, flip_image), axis=1)     # gabungkan wajah kanan kiri            
            else:                                                             # jika lebih luas kanan maka bagian kanan wajah yang dipakai
                half_face = warpp_kanan
                half_face = potong_area_(half_face, landmark_warp_kanan)      # proses memotong hanya bagian wajah
                flip_image = cv2.flip(half_face, 1)                           # flip wajah
                combine = np.concatenate((flip_image, half_face), axis=1)     # gabungkan wajah kanan kanan
            ######################## END - proses membandingkan luas wajah kanan dan kiri #################################################################

            ######################## proses resize & save - START #########################################################################################
            flip_output = combine                                             # flip_output = output proses wajah
            flip_output = cv2.resize(flip_output, (128, 128))                 # resize gambar wajah output
            ######################## END - proses resize & save ##########################################################################################
            return flip_output
    ######################## END - Proses Wajah ######################################################################################################    
        else:
            return None    
    else:
        return None
