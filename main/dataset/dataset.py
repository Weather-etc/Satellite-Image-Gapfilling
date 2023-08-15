import glob
import os
import cv2 as cv
from skimage.morphology import disk, binary_dilation
import numpy as np
import random
from ..tools.mask import build_mask, fetch_land_mask
from ..config import unknown_low, unknown_high

raw_list = glob.glob('../data/dataset_raw/*.tif')
train_dir = '../data/dataset/train'
test_dir = '../data/dataset/test'

mask_shape = (293, 350)


def create_mask(shape, seed):
    """
    Building artificial masks for training set and test set.
    :param shape: shape of masks.
    :param seed: set seeds for random spots in masks.
    :return: mask
    """
    land_mask = fetch_land_mask()

    # Create blocks
    mask = np.zeros(shape, dtype=bool)
    mask[120:140, 70:80] = 1
    mask[100:130, 270:310] = 1
    # Add long, narrow areas
    mask[100:110, 60:120] = 1
    mask[60:100, 150:160] = 1
    # Create spots
    rstate = np.random.default_rng(seed)
    for radius in [0, 2, 4]:
        # larger defects are less common
        thresh = 3 + 0.25 * radius
        tmp_mask = rstate.standard_normal(shape) > thresh
        if radius > 0:
            tmp_mask = binary_dilation(tmp_mask, disk(radius, dtype=bool))
        # only retain spots on water area
        tmp_mask = tmp_mask * ~land_mask
        mask[tmp_mask] = 1
    return mask


def preprocess(images_paths):
    """
    To remove unknown pixels from images, a dilation operation is done.
    Images processed is saved to directory: '/imgTiff/data/dataset'.
    :param images_paths: a list contains paths of images used in function.
    :return: None
    """
    images = map(cv.imread, images_paths)
    num = len(images_paths)
    count = 0
    for img in images:
        # first transform grey area to a small number (int 2)
        img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        img_hs = img[:, :, (0, 1)]
        img_v = img[..., 2]
        dst = build_mask(img, unknown_low, unknown_high)
        img_v = np.where(dst, 2, img_v)

        land_mask = fetch_land_mask()
        img_v_copy = img_v * ~land_mask
        img_v_copy[img_v_copy == 0] = 40

        kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 3), anchor=(1, 0))
        iters = 1
        dilation = cv.dilate(img_v_copy, kernel, iterations=iters)
        dilation_hs = cv.dilate(img_hs, kernel, iterations=iters)
        dilation[dilation == 40] = 0

        res_v = dilation * ~land_mask + img_v * land_mask
        dilation_hs[..., 0] = dilation_hs[..., 0] * ~land_mask + img_hs[..., 0] * land_mask
        dilation_hs[..., 1] = dilation_hs[..., 1] * ~land_mask + img_hs[..., 1] * land_mask

        res_v = res_v[..., np.newaxis]
        res = np.concatenate((dilation_hs, res_v), axis=2)
        res = cv.cvtColor(res, cv.COLOR_HSV2BGR)

        # save images preprocessed to directories
        if count / num <= 0.7:
            path = f'../data/dataset/train/{images_paths[count][-8:]}'
            cv.imwrite(path, res)
        else:
            path = f'../data/dataset/test/{images_paths[count][-8:]}'
            cv.imwrite(path, res)
        count += 1


def merge_save(img_paths, save_path):
    """
    Merge image and its mask together.
    :param img_paths: a list of paths of images
    :param save_path: path where images will be saved to
    :return: a numpy.uint8 numpy.ndarray with the same shape of image provided
    """
    for i in range(len(img_paths)):
        mask = create_mask(mask_shape, i)
        img = cv.imread(img_paths[i])
        img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        # create image with mask
        img_defect_h = np.where(mask == 0, img[..., 0], 0)
        img_defect_s = np.where(mask == 0, img[..., 1], 0)
        img_defect_v = np.where(mask == 0, img[..., 2], 140)
        img_defect = np.zeros(img.shape)
        img_defect[..., 0] = img_defect_h
        img_defect[..., 1] = img_defect_s
        img_defect[..., 2] = img_defect_v
        img_defect = img_defect.astype(np.uint8)
        img_defect = cv.cvtColor(img_defect, cv.COLOR_HSV2BGR)
        # create image of mask
        img_mask = 255 - np.zeros(mask_shape)
        img_mask = np.where(mask == 0, img_mask, 140)

        path = os.path.join(save_path, img_paths[i][-8:])
        path_mask = os.path.join(save_path, img_paths[i][-8:-4] + '_mask.png')
        cv.imwrite(path, img_defect)
        cv.imwrite(path_mask, img_mask)


def construct():
    if not len(os.listdir(train_dir)) or not len(os.listdir(test_dir)):
        random.Random(0).shuffle(raw_list)
        preprocess(raw_list)
    train_img_paths = glob.glob(train_dir + '/*.tif')
    test_img_paths = glob.glob(test_dir + '/*.tif')
    merge_save(train_img_paths, train_dir)
    merge_save(test_img_paths, test_dir)


if __name__ == '__main__':
    construct()
