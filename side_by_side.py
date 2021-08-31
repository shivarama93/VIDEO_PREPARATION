import cv2
import os

image_folder_LR = '/home/shivarama/BasicSR/datasets/Vid4/BIx4/foliage'
image_folder_HR = '/home/shivarama/BasicSR/results/EDVR_L_x4_Vimeo90K_SR_official_On_Vid4_Data/visualization/Vid4/foliage'
video_name = 'out_video.avi'

images_LR = [img for img in os.listdir(image_folder_LR) if img.endswith(".png")]
images_LR = sorted(images_LR,key=lambda x: int(os.path.splitext(x[0:8])[0]))
images_HR = [img for img in os.listdir(image_folder_HR) if img.endswith(".png")]
images_HR = sorted(images_HR,key=lambda x: int(os.path.splitext(x[0:8])[0]))
outframe_HR = cv2.imread(os.path.join(image_folder_HR, images_HR[0]))
outframe_LR = cv2.imread(os.path.join(image_folder_LR, images_LR[0]))
height_HR, width_HR, layers_HR = outframe_HR.shape
height_LR, width_LR, layers_LR = outframe_LR.shape
padsize_hor = int((3*width_LR)/2)
padsize_ver = int((3*height_LR)/2)

video = cv2.VideoWriter(video_name, 0, 15, (width_HR*2,height_HR))
BLACK = [0,0,0]
for idx in range(len(images_LR)):
    image_LR = cv2.imread(os.path.join(image_folder_LR, images_LR[idx]))
    image_HR = cv2.imread(os.path.join(image_folder_HR, images_HR[idx]))
    image_LR_padded = cv2.copyMakeBorder(image_LR.copy(), padsize_ver, padsize_ver, padsize_hor, padsize_hor, cv2.BORDER_CONSTANT, value=BLACK)
    combined = cv2.hconcat([image_LR_padded, image_HR])
    video.write(combined)

cv2.destroyAllWindows()
video.release()
