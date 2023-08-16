import cv2 as cv
import numpy as np

land_path = '../data/dataset/0218.tif'

land_low = np.array([0, 0, 240])
land_high = np.array([0, 0, 256])


def get_mask(img_path):
    """
    This function reads *_mask.png in and transform it into a numpy.ndarray.
    By default, the *_mask.png is under the same path of the image.
    :param img_path: path of image.
    :return: a numpy.ndarray
    """
    mask_path = img_path[:-4] + '_mask.png'
    img = cv.imread(mask_path, flags=0)
    mask = np.array([255, ], dtype=np.uint8) - img
    mask = mask > 0
    return mask


def build_mask(img, range_low, range_high):
    """
    This function will build mask for area specified by parameters.
    Mask will use True for selected pixels and False for unselected area.
    :return: mask - a bool numpy.ndarray shaped (293, 350)
    """
    mask = cv.inRange(img, range_low, range_high) // 255
    mask = mask.astype(bool)
    return mask


def fetch_land_mask():
    """
    This function produces a mask indicating land pixels.
    :return: a bool numpy.ndarray where int 1 configs land while int 0 configs water.
    """
    img = cv.imread(land_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = build_mask(img, land_low, land_high)
    return mask
