import glob
import cv2 as cv
import numpy as np

path_list = glob.glob('../data/rawTif/*.tif')
img_list = map(cv.imread, path_list)

count = 0
count_clean = 0
img_size = 0
water_size = 0
for img in img_list:
    print(path_list[count][-8:])
    try:
        img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        if count == 0:
            lower_water = np.array([0, 0, 0])
            upper_water = np.array([0, 0, 240])
            mask_water = cv.inRange(img_hsv, lower_water, upper_water)
            water_size = len(mask_water.flatten())
        count += 1

        # detect unknown area: set lower bound and upper bound
        # these bound is selected manually with imgVisible.py
        lower = np.array([0, 0, 80])
        upper = np.array([0, 0, 230])
        mask = cv.inRange(img_hsv, lower, upper)

        if len(mask[mask != 0].flatten()) / water_size < 0.05:
            count_clean += 1
            cv.imwrite('../data/cleanImg/{}'.format(path_list[count][-8:]), img)
    except Exception as e:
        pass
print(count_clean)

