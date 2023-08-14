import os.path

import cv2 as cv
import numpy as np
from ..tools.mask import get_mask, fetch_land_mask
from colorama import Fore
from skimage.restoration import inpaint


class InpaintModel:
    def __init__(self, img_paths, save_path):
        self.img_paths = img_paths
        self.save_path = save_path

    def dilate_masked(self, img, mask, kernel):
        """
        This method performs a dilate operation with mask provided.
        :param img: a numpy.uint8 numpy.ndarray. Image needed to do dilation.
        :param mask: a bool numpy.ndarray. Mask specified the area to do dilation.
        :param kernel: a 0-1 numpy.ndarray. Kernel of dilation. Notice that shape must be odd.
        :return: output: a numpy.uint8 numpy.ndarray with the same size of img.
        """
        # ensure inputs have the same dimensions
        if img.ndim != mask.ndim:
            mask = mask.reshape(img.shape)
        if img.ndim > kernel.ndim + 1:
            print(Fore.RED + 'ERROR: dimension of kernel is not match')
            exit(1)
        elif img.ndim > kernel.ndim:
            kernel = kernel[..., np.newaxis]
        else:
            print(Fore.RED + 'ERROR: dimension of image is less than kernel')
            exit(1)
        if kernel.shape[-1] < img.shape[-1]:
            if kernel.shape[-1] == 1:
                ker_new = (kernel, ) * img.shape[-1]
                kernel = np.concatenate(ker_new, axis=-1)
            else:
                print(Fore.RED + 'ERROR: shape of kernel is not match')
                exit(1)

        indices = np.where(mask == 1)
        shape = kernel.shape()
        output = np.zeros(img.shape)
        for index in indices:
            range0 = index - shape // 2
            range1 = index + shape // 2
            submat = img[range0:range1]
            submat = submat * kernel
            res = np.max(submat)
            output[index] = res
        output = output * mask + output * ~mask
        return output

    def morphology(self, img, img_mask, opt):
        """
        Perform initial dilate.
        :param img: a numpy.uint8 numpy.ndarray. Should in HSV format.
        :param img_mask: a bool numpy.ndarray. The mask of img, should be a bool numpy.ndarray.
        :return: res: a numpy.uint8 numpy.ndarray. Image after dilate
        :return mask_new: a bool numpy.ndarray. New mask after morphological operations.
        """
        land_mask = fetch_land_mask()
        img_hs = img[..., (0, 1)]
        img_v = img[..., 2]
        # do some transforms before dilation
        img_v_copy = np.where(img_mask, 2, img_v)
        img_v_copy = img_v_copy * ~land_mask
        # perform morphological operation
        morph_v = img_v_copy
        morph_hs = img_hs
        mask_new = img_mask
        if opt == 'ini_dilate':
            iters = 1
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 3))
            morph_v = cv.dilate(morph_v, kernel, iterations=iters)
            morph_hs = cv.dilate(morph_hs, kernel, iterations=iters)
            mask_new = ~cv.dilate(~img_mask, kernel, iterations=iters)
        elif opt == 'close':
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
            morph_v = cv.morphologyEx(morph_v, cv.MORPH_CLOSE, kernel)
            morph_hs = cv.morphologyEx(morph_hs, cv.MORPH_CLOSE, kernel)
            mask_new = ~cv.morphologyEx(~img_mask, cv.MORPH_CLOSE, kernel)
        elif opt == 'hole_fill':
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
            morph_v = self.dilate_masked(morph_v, img_mask, kernel)
            morph_hs = self.dilate_masked(morph_hs, img_mask, kernel)
            mask_new = ~cv.morphologyEx(~img_mask, img_mask, kernel)
        else:
            print(Fore.RED + 'ERROR: Unsupported morphological operation: ' + opt)
            exit(1)
        # generate result
        res = np.zeros(img.shape)
        res[..., 0] = morph_hs[0] * ~land_mask + img_hs[0] * land_mask
        res[..., 1] = morph_hs[1] * ~land_mask + img_hs[1] * land_mask
        res[..., 2] = morph_v * ~land_mask + img_v * land_mask
        return res, mask_new

    def inpaint_BHM(self, img, img_mask):
        """
        Perform Biharmonic inpainting.
        :param img: a numpy.uint8 numpy.ndarray. Image to do inpainting on.
        :param img_mask: a bool numpy.ndarray. Mask indicating where to do inpainting.
        :return: None
        """
        # TODO: Complete this.
        return img

    def inpaint(self):
        imgs = map(cv.imread, self.img_paths)

        count = 0
        for img in imgs:
            img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            img_mask = get_mask(self.img_paths[count])
            # Perform initial dilation
            img, img_mask = self.morphology(img, img_mask, 'ini_dilate')
            # do small hole closure
            img, img_mask = self.morphology(img, img_mask, 'close')
            # small hole fill
            img, img_mask = self.morphology(img, img_mask, 'hole_fill')

            # if unknown pixels still exist:
            if len(np.unique(img_mask)) > 1:
                img = self.inpaint_BHM(img, img_mask)

            img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
            path = os.path.join(self.save_path, self.img_paths[count][-8:-4] + '_res.tif')
            cv.imwrite(path, img)
            count += 1
        print('inpainting completed')
