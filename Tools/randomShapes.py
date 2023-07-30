import matplotlib.pyplot as plt
from skimage.draw import random_shapes
import cv2 as cv
import numpy as np

result = random_shapes((293, 350), max_shapes=1, shape='rectangle',
                       channel_axis=None, intensity_range=(100, 100),
                       max_size=100, min_size=30)
mask, _ = result
mask_array = np.array(mask)
print('shape: ', mask_array.shape)
mask_array[mask_array == 100] = 0
mask_array[mask_array != 100] = 255
print(mask_array)

with open('../data/cleanImg/0124.tif', 'w', encoding='utf-8') as f:
    img =