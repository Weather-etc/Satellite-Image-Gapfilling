"""
This file aimed at selecting clear satellite images
Note:
    For HSV, hue range is [0,179], saturation range is [0,255], and value range is [0,255].
    Refer to OpenCV tutorial (https://docs.opencv.org/4.8.0/df/d9d/tutorial_py_colorspaces.html)
"""
import cv2
import numpy as np

print('part0')
img_0 = cv2.imread('../data/rawTif/0105.tif')
cv2.imshow("raw image", img_0)

hsv_low = np.array([0, 0, 0])
hsv_high = np.array([0, 0, 0])


def h_low(value):
    hsv_low[0] = value


def h_high(value):
    hsv_high[0] = value


def s_low(value):
    hsv_low[1] = value


def s_high(value):
    hsv_high[1] = value


def v_low(value):
    hsv_low[2] = value


def v_high(value):
    hsv_high[2] = value


cv2.namedWindow("modify")
cv2.createTrackbar('H low', 'modify', 0, 179, h_low)
cv2.createTrackbar('H high', 'modify', 0, 179, h_high)
cv2.createTrackbar('S low', 'modify', 0, 255, s_low)
cv2.createTrackbar('S high', 'modify', 0, 255, s_high)
cv2.createTrackbar('V low', 'modify', 0, 255, v_low)
cv2.createTrackbar('V high', 'modify', 0, 255, v_high)
while True:
    dst = cv2.cvtColor(img_0, cv2.COLOR_BGR2HSV) # 转HSV
    dst = cv2.inRange(dst, hsv_low, hsv_high) # 通过HSV的高低阈值，提取图像部分区域
    cv2.imshow('dst', dst)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

