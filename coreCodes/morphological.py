import cv2
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.morphology import disk
from skimage.restoration import inpaint
from scipy.ndimage import binary_dilation

imgpath = "../Exemplar/tests/Mur/0718_test_resultat.jpg"


def generate_mask(img):
    # Create mask with six block defect regions
    mask = np.zeros(img.shape[:-1], dtype=bool)
    mask[20:60, 0:20] = 1
    mask[160:180, 70:155] = 1
    mask[30:60, 170:195] = 1
    mask[-60:-30, 170:195] = 1
    mask[-180:-160, 70:155] = 1
    mask[-60:-20, 0:20] = 1

    # add a few long, narrow defects
    mask[200:205, -200:] = 1
    mask[150:255, 20:23] = 1
    mask[365:368, 60:130] = 1

    # add randomly positioned small point-like defects
    rstate = np.random.default_rng(0)
    for radius in [0, 2, 4]:
        # larger defects are less common
        thresh = 3 + 0.25 * radius  # make larger defects less common
        tmp_mask = rstate.standard_normal(img.shape[:-1]) > thresh
        if radius > 0:
            tmp_mask = binary_dilation(tmp_mask, disk(radius, dtype=bool))
        mask[tmp_mask] = 1
    return mask


def img_show(img, title):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.show()


def fft(img):
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    return dft_shift


def image_dilation(imgpath):
    img = cv.imread(imgpath)
    # kernel = cv.getStructuringElement(cv.MORPH_DILATE, (6, 6), (5, 5))
    kernel = np.ones((3, 3))
    erosion = cv.erode(img, kernel, iterations=2)
    kernel1 = np.ones((3, 3))
    dilation = cv.dilate(erosion, kernel1, iterations=5)
    cv.imshow('dilation', dilation)
    cv.waitKey(0)


def main():
    img = cv.imread('../data/biharmonic.png', 1)
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    img_v = img_hsv[..., 2]

    grad_x = cv.Sobel(img_v, cv.CV_64F, 2, 0)
    grad_y = cv.Sobel(img_v, cv.CV_64F, 0, 2)
    grad = cv.addWeighted(grad_y, 0.5, grad_x, 0.5, 0)
    plt.imshow(grad, cmap='gray')
    plt.show()

    fft_grad = fft(grad)

    # gauss_kersize = 3
    # img_v = cv.GaussianBlur(img_v, (gauss_kersize, gauss_kersize), 0, 0)
    # cv.imshow('blurred', img_v)

    img2 = cv.imread('../data/biharmonic.png', 1)
    img2_h = img2[..., 0]
    img2_s = img2[..., 1]
    img2_v = img_hsv[..., 2]
    fft2_v = fft(img2_v)
    bihar_v = img2_v

    '''img_show(img_v, 'img_v')
    img_show(img_h, 'img_h')
    img_show(img_s, 'img_s')
    img_show(fft_v, 'fft result on img_v')
    img_show(fft_h, 'fft_h')
    img_show(fft_s, 'fft_s')'''

    rows, cols = img_v.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), bool)
    mask[crow-40:crow+40, ccol-30:ccol+30] = 0
    ifft_v = fft2_v * ~mask + fft_grad * mask

    ifft_v = np.fft.ifftshift(ifft_v)
    ifimage_v = np.fft.ifft2(ifft_v)
    ifimage_hs = np.abs(ifimage_v)

    plt.imshow(ifimage_hs, cmap='gray')
    plt.title("dft_image_v")
    plt.show()
    plt.imshow(bihar_v, cmap='gray')
    plt.title("biharmonic v")
    plt.show()

    img_final = np.array((img2_h, img2_s, ifimage_hs)).reshape(img.shape)
    print(img.shape)
    # img_final = cv.cvtColor(img_final, cv.COLOR_HSV2BGR)
    plt.imshow(img_final)
    plt.show()

    """
    mask = generate_mask(img)

    # apply mask to image
    img_masked = img * ~mask[..., np.newaxis]
    plt.imshow(img_masked)
    plt.title("origin image")
    plt.show()

    img_used = img_masked[mask == 1]
    print(img_used.size)
    plt.imshow(img_used)
    plt.title("img_used")
    plt.show()

    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (6, 6), (0, 0))
    print(kernel)
    erosion = cv.erode(img_used, kernel, iterations=1)
    dilation = cv.dilate(erosion, kernel, iterations=1)

    img_masked[mask == 1] = dilation
    plt.imshow(img_masked)
    plt.title("img after morphological operation")
    plt.show()
    """


if __name__ == '__main__':
    main()
