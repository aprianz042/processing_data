from frontalization import *

file_ = 'tangan2.jpg'
img = cv2.imread(file_)
print(img.shape)
flow, ori, output_image = half_flip(img)
show_images_grid(flow)
show_img(ori)


