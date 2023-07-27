import numpy as np
import seaborn as sns
import glob
import cv2 as cv
import matplotlib.pyplot as plt

path_list = glob.glob('../data/rawTif/*.tif')
img_list = map(cv.imread, path_list)

bloom_count = np.zeros((293, 350))
for img in img_list:
    try:
        img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)

        lower_bloom = np.array([0, 0, 0])
        upper_bloom = np.array([0, 0, 255])

        mask_bloom = cv.inRange(img_hsv, lower_bloom, upper_bloom)
        bloom_count[mask_bloom == 0] += 1
    except Exception as e:
        pass
heatmap = sns.heatmap(bloom_count)
heatmap.set_title('times of bloom occurred')
plt.show()
figure = heatmap.get_figure()
figure.savefig('../data/times_heatmap.jpg')
