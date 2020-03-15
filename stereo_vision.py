from stereovision import blockmatchers, stereo_cameras, ui_utils, calibration
import cv2


if __name__ == '__main__':
    ui_utils.BMTuner(blockmatchers.StereoBM())
    sp = stereo_cameras.StereoPair([0,2])
    frame1, frame2 = sp.get_frames()
    bn = blockmatchers.StereoSGBM()
    cv2.imshow("disp", bn.get_disparity([frame1, frame2]))
    cv2.waitKey()
    cv2.destroyAllWindows()
