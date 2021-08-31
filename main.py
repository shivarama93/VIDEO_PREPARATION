import cv2
import os

def sideBySide(image_folder_LR=None, image_folder_HR=None, video_name='out_video.avi'):
    if image_folder_LR == None or image_folder_HR == None:
        image_folder_LR = '/home/shivarama/BasicSR/datasets/Vid4/BIx4/foliage'
        image_folder_HR = '/home/shivarama/BasicSR/results/EDVR_L_x4_Vimeo90K_SR_official_On_Vid4_Data/visualization/Vid4/foliage'

    images_LR = getFilenamesSorted(image_folder=image_folder_LR)
    images_HR = getFilenamesSorted(image_folder=image_folder_HR)

    outframe_HR = cv2.imread(os.path.join(image_folder_HR, images_HR[0]))
    outframe_LR = cv2.imread(os.path.join(image_folder_LR, images_LR[0]))
    height_HR, width_HR, layers_HR = outframe_HR.shape
    height_LR, width_LR, layers_LR = outframe_LR.shape
    padsize_hor = int((3 * width_LR) / 2)
    padsize_ver = int((3 * height_LR) / 2)

    video = cv2.VideoWriter(video_name, 0, 15, (width_HR * 2, height_HR))
    BLACK = [0, 0, 0]
    for idx in range(len(images_LR)):
        image_LR = cv2.imread(os.path.join(image_folder_LR, images_LR[idx]))
        image_HR = cv2.imread(os.path.join(image_folder_HR, images_HR[idx]))
        image_LR_padded = cv2.copyMakeBorder(image_LR.copy(), padsize_ver, padsize_ver, padsize_hor, padsize_hor,
                                             cv2.BORDER_CONSTANT, value=BLACK)
        combined = cv2.hconcat([image_LR_padded, image_HR])
        video.write(combined)

    cv2.destroyAllWindows()
    video.release()

def getFilenamesSorted(image_folder=None):
    if image_folder is None:
        return
    filelist = [img for img in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, img))]
    pinch_of_salt = filelist[0:20] if len(filelist) >= 20 else filelist
    commonprefix = os.path.commonprefix(pinch_of_salt)
    commonsuffix = os.path.commonprefix([x[::-1] for x in pinch_of_salt])
    commonsuffix = commonsuffix[::-1]
    sorted_list = sorted(filelist, key=lambda x: int(os.path.splitext(x[len(commonprefix):x.rfind(commonsuffix)])[0]))
    return sorted_list

def sideBySideWithScaled(image_folder_LR=None, image_folder_HR=None, video_name='out_video.avi'):
    if image_folder_LR == None or image_folder_HR == None:
        print("No imgages.. nothing to do!!")
        return

    images_LR = getFilenamesSorted(image_folder=image_folder_LR)
    images_HR = getFilenamesSorted(image_folder=image_folder_HR)

    outframe_HR = cv2.imread(os.path.join(image_folder_HR, images_HR[0]))
    height_HR, width_HR, layers_HR = outframe_HR.shape

    video = cv2.VideoWriter(video_name, 0, 15, (width_HR * 2, height_HR))
    for idx in range(len(images_HR)):
        image_LR = cv2.imread(os.path.join(image_folder_LR, images_LR[idx]))
        image_HR = cv2.imread(os.path.join(image_folder_HR, images_HR[idx]))
        image_LR_resized = cv2.resize(image_LR.copy(), (width_HR, height_HR))
        combined = cv2.hconcat([image_LR_resized, image_HR])
        video.write(combined)

    cv2.destroyAllWindows()
    video.release()

def threeSideBySideWithScaled(image_folder_1=None, image_folder_2=None, fixed_scale=False,
                              image_folder_3=None, frame_rate=15, video_name='out_video.avi'):
    if image_folder_1 == None or image_folder_2 == None or image_folder_3 == None:
        print("No imgages.. nothing to do!!")
        return

    images_1 = getFilenamesSorted(image_folder=image_folder_1)
    images_2 = getFilenamesSorted(image_folder=image_folder_2)
    images_3 = getFilenamesSorted(image_folder=image_folder_3)

    if fixed_scale is True:
        width_HR, height_HR = 640, 480
    else:
        outframe_HR = cv2.imread(os.path.join(image_folder_3, images_3[0]))
        height_HR, width_HR, layers_HR = outframe_HR.shape


    video = cv2.VideoWriter(video_name, 0, frame_rate, (width_HR * 3, height_HR))
    for idx in range(len(images_3)):
        image_1 = cv2.imread(os.path.join(image_folder_1, images_1[idx]))
        image_2 = cv2.imread(os.path.join(image_folder_2, images_2[idx]))
        image_3 = cv2.imread(os.path.join(image_folder_3, images_3[idx]))
        image_1_resized = cv2.resize(image_1.copy(), (width_HR, height_HR))
        image_2_resized = cv2.resize(image_2.copy(), (width_HR, height_HR))
        image_3_resized = cv2.resize(image_3.copy(), (width_HR, height_HR))
        combined = cv2.hconcat([image_1_resized, image_2_resized, image_3_resized])
        video.write(combined)

    cv2.destroyAllWindows()
    video.release()

def findFramerate(video = None):
    if video == None:
        video = cv2.VideoCapture("/home/shivarama/BasicSR/datasets/SK_POC/01_SeriPark_Input_Re_x2.mp4")
    fps = video.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    video.release()

def dumpLRHRFramesFromVideo(vid_file_name=None,
                            folder_prefix='OUTPUT',
                            num_of_frames=None,
                            scale=4):
    # Opens the Video file
    if vid_file_name == None:
        print('Nothing to do. No input!!')
        return

    cap = cv2.VideoCapture(vid_file_name)

    if num_of_frames == None:
        num_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("\nNumber of frames unspecified, taking all frames, {} in total.".format(num_of_frames))

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

def dumpFramesFromVideo(vid_file_name=None,
                        num_of_frames=None,
                        output_image_size=None,
                        framegap=1,
                        out_folder='OUTPUT'):

    if vid_file_name == None:
        return

    if '\\' in vid_file_name:
        vid_file_name = vid_file_name.replace('\\' , '/')

    cap = cv2.VideoCapture(vid_file_name)
    if num_of_frames == None:
        num_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("\nNumber of frames unspecified, taking all frames, {} in total.".format(num_of_frames))

    target_foldername = vid_file_name[vid_file_name.rfind('/') + 1:vid_file_name.rfind('.')]
    target_fullpath = os.path.join(out_folder, target_foldername)
    if not os.path.exists(target_fullpath):
        os.mkdir(target_fullpath)

    i = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False or i == num_of_frames:
            break
        i += 1
        if i % framegap == 0:
            if output_image_size is not None:
                out_image = cv2.resize(frame, output_image_size, interpolation=cv2.INTER_AREA)
            else:
                out_image = frame
            cv2.imwrite(os.path.join(target_fullpath,  target_foldername + "FN{}.png".format(i)), out_image)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # dumpFramesFromVideoGeneral(vid_file_name="D:/data/SKT_Project/SK_POC/03_A.mp4")
    # dumpFramesFromVideo(vid_file_name = '/home/shivarama/BasicSR/datasets/SK_POC/01_SeriPark_Input_Re_x2.mp4',
    #                     folder_prefix = '/home/shivarama/BasicSR/datasets/test/Vid4',
    #                     num_of_frames=100)

    # sideBySideWithScaled(image_folder_HR = r'D:\PYTHON_WORK_ENV\IMAGE_RESTORATION_JULY_2021\TESTING_OUTPUT\deoldify_19_08_2021_14_35_54',
    #                      image_folder_LR = r'D:\data\SKT_Project\video_frames\03_A',
    #                      video_name='OUTPUT/03_A_Restored_full.mp4')

    # dumpFramesFromVideo(vid_file_name=r"D:\data\SKT_Project\210616_Test\01_국가기록물\국가기록원_B.mp4",
    #                     framegap=1,
    #                     out_folder='output/test_frames')

    threeSideBySideWithScaled(image_folder_1=r'D:\data\SKT_Project\video_frames_filtered\SELECTED\03_A',
                              image_folder_2=r'D:\PYTHON_WORK_ENV\Real-SR\results\03A\DItest_SKT.ymlV2K',
                              image_folder_3=r'D:\PYTHON_WORK_ENV\BasicSR\results\EDVR_L_skt_deblur\visualization\REDS4\03_A',
                              frame_rate=1,
                              fixed_scale=True,
                              video_name='OUTPUT/Comparison_RealSR_EDVR_03A_new.mp4')
