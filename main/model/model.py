import os
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tools.mask import get_mask, fetch_land_mask
from colorama import Fore
from skimage.restoration import inpaint


class InpaintModel:
    def __init__(self, land_mask):
        self.DILATE = 0
        self.ERODE = 1
        self.CLOSE = 2
        self.land_mask = land_mask

    @staticmethod
    def image_padding(img, shape):
        """
        Do image padding according to shape of kernel.
        :param img: a numpy.ndarray. Image need to be padded.
        :param shape: a numpy.ndarray. Shape of kernel.
        :return: img_pad: a numpy.ndarray. Padded image
        """
        if img.ndim == 3:
            img_pad = np.pad(img, ((shape[0] // 2, shape[0] // 2),
                                   (shape[1] // 2, shape[1] // 2),
                                   (0, 0)),
                             'constant', constant_values=0)
        elif img.ndim == 2:
            img_pad = np.pad(img, ((shape[0] // 2, shape[1] // 2),
                                   (shape[1] // 2, shape[1] // 2)),
                             'constant', constant_values=0)
        else:
            raise ValueError('The dimension of image is not allowed')
        return img_pad

    @staticmethod
    def adjust_kernel(img, kernel):
        """
        This method will adjust kernel to have the same dimensions as image.
        :param img: a numpy.ndarray. Image the kernel will be used on.
        :param kernel: a numpy.ndarray. Kernel to used on image.
        :return: a numpy.ndarray. Adjusted kernel.
        """
        if kernel.ndim == img.ndim:
            return kernel
        # Check whether diemsions of image and kernel match
        if img.ndim > kernel.ndim + 1:
            raise ValueError('dimensions of kernel and image are not match')
        elif img.ndim > kernel.ndim:
            kernel = kernel[..., np.newaxis]
        else:
            raise ValueError('dimension of image is less than kernel')
        # Expand kernel to different channels
        if kernel.shape[-1] < img.shape[-1]:
            if kernel.shape[-1] == 1:
                ker_new = (kernel, ) * img.shape[-1]
                kernel = np.concatenate(ker_new, axis=-1)
            else:
                raise ValueError('shape of kernel is not match')
        return kernel

    def morph_masked_opt(self, img, mask, kernel, opt):
        """

        :param img:
        :param mask:
        :param kernel:
        :param opt:
        :return:
        """
        kernel = self.adjust_kernel(img, kernel)
        indices = np.where(mask == 1)
        indices = zip(*indices)
        indices = map(np.array, indices)
        shape = kernel.shape[:2]
        output = np.zeros(img.shape)
        img_pad = self.image_padding(img, shape)

        for index in indices:
            index_pad = index + np.array(shape) // 2
            range0 = np.array(index_pad - np.array(shape) // 2)
            range0 = range0.astype(np.int16)
            range1 = np.array(index_pad + np.array(shape) // 2 + 1)
            range1 = range1.astype(np.int16)

            submat = img_pad[range0[0]:range1[0], range0[1]:range1[1]].copy()
            submat = submat * kernel
            res = np.max(submat) if opt == 'dilate' else np.min(submat)
            output[tuple(index)] = res

        output = output * mask + img * ~mask
        output = output.astype(np.uint8)
        return output

    def morph_masked(self, img, mask, kernel, opt):
        """
        This method performs a dilate operation with mask provided.
        :param img: a numpy.uint8 numpy.ndarray. Image needed to do dilation.
        :param mask: a bool numpy.ndarray. Mask specified the area to do dilation.
        :param kernel: a 0-1 numpy.ndarray. Kernel of dilation. Notice that shape must be odd.
        :param opt: int.
        :return: output: a numpy.uint8 numpy.ndarray with the same size of img.
        """
        if opt not in (self.DILATE, self.CLOSE):
            raise ValueError("Unsupported operation " + opt)
        if kernel.ndim < img.ndim:
            res = [None, ] * img.shape[-1]
            for i in range(img.shape[-1]):
                res[i] = self.morph_masked_opt(img[..., i].copy(), mask, kernel, opt)
                res[i] = res[i][..., np.newaxis]
            output = np.concatenate(res, axis=-1)
            print(output.shape)
        elif kernel.ndim == img.ndim:
            output = self.morph_masked_opt(img.copy(), mask, kernel, opt)
        else:
            raise ValueError('dimensions of kernel and image do not match')
        return output

    def morph_operate(self, morph_v, morph_hs, mask, opt):
        """
        Do morphological operations on channel V and HS respectively.
        :param morph_v: a numpy.uint8 numpy.ndarray.
        :param morph_hs: a numpy.uint8 numpy.ndarray.
        :param mask: a numpy.uint8 numpy.ndarray
        :param opt: str. Flags of operation to be performed.
        :return: morph_v, morph_hs, mask. All are numpy.uint8 numpy.ndarray.
        """
        if opt == 'ini_dilate':
            iters = 1
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 3))
            morph_v = cv.dilate(morph_v, kernel, iterations=iters)
            morph_hs = cv.dilate(morph_hs, kernel, iterations=iters)
            mask = ~cv.dilate(~mask, kernel, iterations=iters)
        elif opt == 'close':
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
            morph_v = cv.dilate(morph_v, kernel, iterations=1)
            morph_v = cv.erode(morph_v, kernel, iterations=1)
            morph_hs = cv.dilate(morph_hs, kernel, iterations=1)
            morph_hs = cv.dilate(morph_hs, kernel, iterations=1)
            mask = ~cv.dilate(~mask, kernel)
            mask = ~cv.erode(~mask, kernel)
        elif opt == 'hole_fill':
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
            morph_v = self.morph_masked(morph_v, mask, kernel, self.DILATE)
            morph_hs = self.morph_masked(morph_hs, mask, kernel, self.DILATE)
            mask = ~self.morph_masked(~mask, mask, kernel, self.DILATE)
        else:
            raise ValueError('Unsupported morphological operation: ' + opt)
        return morph_v, morph_hs, mask

    def morphology(self, img, img_mask, opt):
        """

        :param img: a numpy.uint8 numpy.ndarray. Should in HSV format.
        :param img_mask: a bool numpy.ndarray. The mask of img, should be a bool numpy.ndarray.
        :return: res: a numpy.uint8 numpy.ndarray. Image after dilate
        :return mask_new: a bool numpy.ndarray. New mask after morphological operations.
        """
        land_mask = fetch_land_mask(self.land_mask)
        img_hs = img[..., (0, 1)]
        img_v = img[..., 2]
        # do some transforms before dilation
        # ensure the masked area will always be reduced
        img_v_copy = np.where(img_mask, -1, img_v)
        img_v_copy = img_v_copy * ~land_mask
        # perform morphological operation
        morph_v = img_v_copy
        morph_hs = img_hs
        new_mask = img_mask.astype(np.uint8)
        morph_v, morph_hs, new_mask = self.morph_operate(morph_v, morph_hs, new_mask, opt)
        # generate result
        res = np.zeros(img.shape)
        new_mask = new_mask.astype(bool)
        res[..., 0] = morph_hs[..., 0] * ~land_mask + img_hs[..., 0] * land_mask
        res[..., 1] = morph_hs[..., 1] * ~land_mask + img_hs[..., 1] * land_mask
        morph_v[morph_v == -1] = 140
        res[..., 2] = morph_v * ~land_mask + img_v * land_mask
        res = res.astype(np.uint8)
        return res, new_mask

    def inpaint_BHM(self, img, img_mask):
        """
        Perform Biharmonic inpainting.
        :param img: a numpy.uint8 numpy.ndarray. Image to do inpainting on.
        :param img_mask: a bool numpy.ndarray. Mask indicating where to do inpainting.
        :return: None
        """
        # TODO: Complete this.
        land_mask = fetch_land_mask()
        img = land_mask * ~land_mask
        return img

    def inpaint(self, img, img_mask):
        """

        :param img: a numpy.uint8 numpy.ndarray. It should be in HSV format.
        :param img_mask: a bool numpy.ndarray.
        :return: img: a numpy.uint8 numpy.ndarray. Image after inpainting.
        """
        img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        # Perform initial dilation
        # img, img_mask = self.morphology(img, img_mask, 'ini_dilate')
        # do small hole closure
        # img, img_mask = self.morphology(img, img_mask, 'close')
        # masked dilation
        # img, img_mask = self.morphology(img, img_mask, 'hole_fill')
        # if unknown pixels still exist:
        if len(np.unique(img_mask)) > 1:
            land_mask = fetch_land_mask(self.land_mask)
            land_mask = np.concatenate([land_mask[..., np.newaxis], ]*3, axis=-1)
            img_copy = img * ~land_mask
            img_copy = img_copy.astype(np.uint8)

            # especially for cv.inpaint
            # img_mask = img_mask.astype(np.uint8)
            # img_mask = img_mask * 255

            # img_copy = cv.inpaint(img_copy, img_mask, 3, cv.INPAINT_TELEA)
            # img_copy = cv.inpaint(img_copy, img_mask, 3, cv.INPAINT_NS)

            img_copy = inpaint.inpaint_biharmonic(image=img_copy, mask=img_mask, channel_axis=-1)
            img_copy = img_copy * 255
            img = img * land_mask + img_copy * ~land_mask
        img = img.astype(np.uint8)
        img = cv.cvtColor(img, cv.COLOR_HSV2BGR)

        cv.imshow('res', img)
        cv.waitKey(0)

        return img


if __name__ == '__main__':
    path = '../data/dataset/train/0718.tif'
    img0 = cv.imread(path)
    mask = get_mask('../data/dataset/train/0718.tif')
    model = InpaintModel('../data/dataset/0218.tif')
    model.inpaint(img0, mask)
