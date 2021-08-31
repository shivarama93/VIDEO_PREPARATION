import cv2

if __name__ == '__main__':
    video = cv2.VideoCapture("/home/shivarama/BasicSR/datasets/SK_POC/01_SeriPark_Input_Re_x2.mp4")
    fps = video.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    video.release()
