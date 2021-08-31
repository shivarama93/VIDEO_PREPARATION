import cv2
import os

def sideBySide(image_folder_LR=None, image_folder_HR=None, video_name='out_video.avi'):
    if image_folder_LR == None or image_folder_HR == None:
        image_folder_LR = '/home/shivarama/BasicSR/datasets/Vid4/BIx4/foliage'
        image_folder_HR = '/home/shivarama/BasicSR/results/EDVR_L_x4_Vimeo90K_SR_official_On_Vid4_Data/visualization/Vid4/foliage'

    images_LR = [img for img in os.listdir(image_folder_LR) if img.endswith(".jpg")]
    images_LR = sorted(images_LR, key=lambda x: int(os.path.splitext(x[1:])[0]))
    images_HR = [img for img in os.listdir(image_folder_HR) if img.endswith(".tif")]
    images_HR = sorted(images_HR, key=lambda x: int(os.path.splitext(x[1:])[0]))
    outframe_HR = cv2.imread(os.path.join(image_folder_HR, images_HR[0]))
    outframe_LR = cv2.imread(os.path.join(image_folder_LR, images_LR[0]))
    height_HR, width_HR, layers_HR = outframe_HR.shape
    height_LR, width_LR, layers_LR = outframe_LR.shape

    video = cv2.VideoWriter(video_name, 0, 10, (width_HR * 2, height_HR))
    for idx in range(len(images_LR)):
        image_LR = cv2.imread(os.path.join(image_folder_LR, images_LR[idx]))
        image_HR = cv2.imread(os.path.join(image_folder_HR, images_HR[idx]))

        combined = cv2.hconcat([image_LR, image_HR])
        video.write(combined)

    cv2.destroyAllWindows()
    video.release()

def findFramerate(video = None):
    if video == None:
        video = cv2.VideoCapture("/home/shivarama/BasicSR/datasets/SK_POC/01_SeriPark_Input_Re_x2.mp4")
    fps = video.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    video.release()

def dumpFramesFromVideo(vid_file_name=None, folder_prefix='/home/shivarama/BasicSR/datasets/test', num_of_frames=100):
    # Opens the Video file
    if vid_file_name == None:
        vid_file_name = '/home/shivarama/BasicSR/datasets/SK_POC/04_B.mp4'

    scale = 4
    cap = cv2.VideoCapture(vid_file_name)
    target_foldername = vid_file_name[vid_file_name.rfind('/') + 1:vid_file_name.rfind('.')]
    if not os.path.exists(folder_prefix + '/BIx4/' + target_foldername):
        os.mkdir(folder_prefix + '/BIx4/' + target_foldername)
    if not os.path.exists(folder_prefix + '/GT/' + target_foldername):
        os.mkdir(folder_prefix + '/GT/' + target_foldername)
    i = 0
    dsize = None
    while (cap.isOpened()):
        ret, frame = cap.read()
        if dsize is None:
            dsize = (frame.shape[1] * scale, frame.shape[0] * scale)
        if ret == False or i == num_of_frames:
            break
        cv2.imwrite(folder_prefix + '/BIx4/' + target_foldername + "/{:0>8d}".format(i) + '.png', frame)
        GT = cv2.resize(frame, dsize, interpolation=cv2.INTER_AREA)
        cv2.imwrite(folder_prefix + '/GT/' + target_foldername + "/{:0>8d}".format(i) + '.png', GT)
        i += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # dumpFramesFromVideo(vid_file_name = '/home/shivarama/BasicSR/datasets/SK_POC/01_SeriPark_Input_Re_x2.mp4',
    #                     folder_prefix = '/home/shivarama/BasicSR/datasets/test/Vid4',
    #                     num_of_frames=100)

    sideBySide(image_folder_LR = '/home/arshiana/V2/',
               image_folder_HR = '/home/shivarama/Drama2/',
               video_name='OUT_VIDEOS/Drama2.avi')