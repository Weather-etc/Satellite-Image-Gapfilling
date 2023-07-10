"""
This is used to test if K-Means partition is suitable for dataset.
"""
import cv2
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

pathList = glob.glob('./data/rawTif/*.tif')
imgList = map(cv2.imread, pathList)

count = 0
compareGroup = []
for img in imgList:
    imgState = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgReshaped = imgState.reshape(-1, 3)
    imgReshaped = np.float32(imgReshaped)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    attempts = 10
    for K in range(2, 5):
        # Do kmeans. Label is the class for every point on image
        ret, label, center = cv2.kmeans(imgReshaped, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()].reshape(img.shape)
        # use K = 3 to compare results from two image
        if K == 3:
            compareGroup.append(label.flatten())
            print(img.shape)
            print(label.flatten().reshape(img.shape[:-1]))
        savePath = './data/segResult/{}_{}-Means.jpg'.format(pathList[count][15:-4], str(K))
        cv2.imwrite(savePath, res)
    count += 1

label0 = compareGroup[0]
label1 = compareGroup[1]
diffPixel = 0
for i in range(len(label0)):
    if (label1[i] == 0 and label0[i] != 2) or (label1[i] != 0 and label0[i] == 2):
        diffPixel += 1
print('different ratio: ', diffPixel/len(label0))
print('absolute pixels: ', diffPixel)
