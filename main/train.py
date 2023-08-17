import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt
from model.model import InpaintModel
from tools.mask import get_mask


train_dir = "./data/dataset/train/"
test_dir = "./data/dataset/test/"

train_img_paths = glob.glob(train_dir + '*.tif')
test_img_paths = glob.glob(test_dir + '*.tif')
train_imgs = map(cv.imread, train_img_paths)
test_imgs = map(cv.imread, test_img_paths)

count = 0
mse_sum = 0
model = InpaintModel()
for img in train_imgs:
    print(np.max(img))
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    img_path = train_img_paths[count]
    img_mask = get_mask(img_path)
    # perform inpainting and compute mse
    res = model.inpaint(img, img_mask)
    mse = np.square(np.subtract(img, res)).mean()
    mse_sum += mse
    # save predicted-truth pixels value
    pred = res[img_mask == 1]
    truth = img[img_mask == 1]
    pairs = zip(pred, truth)
    # draw a scatter
    value_v_x = []
    value_v_y = []
    for pair in pairs:
        value_v_x.append(pair[0][2])
        value_v_y.append([1][2])

    print(list(pairs))

    count += 1
mse_avg = mse_sum / count
