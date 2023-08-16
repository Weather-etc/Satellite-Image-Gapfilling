import os
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tools.mask import get_mask, fetch_land_mask
from colorama import Fore
from skimage.restoration import inpaint


class InpaintModel:
    def __init__(self, img_paths, save_path):
        self.img_paths = img_paths
        self.save_path = save_path

    @staticmethod
    def image_padding(img, shape):
        """
        Do image padding according to shape of kernel.
        :param img: a numpy.ndarray. Image need to be padded.
        :param shape: a numpy.ndarray. Shape of kernel.
        :return: img_pad: a numpy.ndarray. Padded image
        """
        img_pad = img
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
            print(Fore.RED + 'ERROR: Unsupported image dimension')
            exit(1)
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
            print(Fore.RED + 'ERROR: dimension of kernel is not match')
            exit(1)
        elif img.ndim > kernel.ndim:
            kernel = kernel[..., np.newaxis]
        else:
            print(Fore.RED + 'ERROR: dimension of image is less than kernel')
            exit(1)
        # Expand kernel to different channels
        if kernel.shape[-1] < img.shape[-1]:
            if kernel.shape[-1] == 1:
                ker_new = (kernel, ) * img.shape[-1]
                kernel = np.concatenate(ker_new, axis=-1)
            else:
                print(Fore.RED + 'ERROR: shape of kernel is not match')
                exit(1)
        return kernel

    def dilate_masked(self, img, mask, kernel):
        """
        This method performs a dilate operation with mask provided.
        :param img: a numpy.uint8 numpy.ndarray. Image needed to do dilation.
        :param mask: a bool numpy.ndarray. Mask specified the area to do dilation.
        :param kernel: a 0-1 numpy.ndarray. Kernel of dilation. Notice that shape must be odd.
        :return: output: a numpy.uint8 numpy.ndarray with the same size of img.
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

            submat = img_pad[range0[0]:range1[0], range0[1]:range1[1]]
            submat = submat * kernel
            res = np.max(submat)
            output[tuple(index)] = res

        if mask.ndim < img.ndim:
            mask = mask[..., np.newaxis]
            mask_new = (mask, ) * img.shape[-1]
            mask = np.concatenate(mask_new, axis=2)
        output = output * mask + img * ~mask
        output = output.astype(np.uint8)
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
        # ensure the masked area will always be reduced
        img_v_copy = np.where(img_mask, -1, img_v)
        img_v_copy = img_v_copy * ~land_mask
        # perform morphological operation
        morph_v = img_v_copy
        morph_hs = img_hs
        new_mask = img_mask.astype(np.uint8)
        if opt == 'ini_dilate':
            iters = 1
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 3))
            morph_v = cv.dilate(morph_v, kernel, iterations=iters)
            morph_hs = cv.dilate(morph_hs, kernel, iterations=iters)
            new_mask = ~cv.dilate(~new_mask, kernel, iterations=iters)
        elif opt == 'close':
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
            morph_v = cv.dilate(morph_v, kernel, iterations=1)
            morph_v = cv.erode(morph_v, kernel, iterations=1)
            morph_hs = cv.dilate(morph_hs, kernel, iterations=1)
            morph_hs = cv.dilate(morph_hs, kernel, iterations=1)
            new_mask = ~cv.dilate(~new_mask, kernel)
            new_mask = ~cv.erode(~new_mask, kernel)
            # morph_v = cv.morphologyEx(morph_v, cv.MORPH_CLOSE, kernel)
            # morph_hs = cv.morphologyEx(morph_hs, cv.MORPH_CLOSE, kernel)
            # new_mask = ~cv.morphologyEx(~new_mask, cv.MORPH_CLOSE, kernel)
        elif opt == 'hole_fill':
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
            morph_v = self.dilate_masked(morph_v, img_mask, kernel)
            morph_hs = self.dilate_masked(morph_hs, img_mask, kernel)
            new_mask = ~self.dilate_masked(~new_mask, img_mask, kernel)
        else:
            print(Fore.RED + 'ERROR: Unsupported morphological operation: ' + opt)
            exit(1)
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

    def inpaint(self):
        imgs = map(cv.imread, self.img_paths)

        count = 0
        for img in imgs:
            img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            img_mask = get_mask(self.img_paths[count])
            cv.waitKey(0)

            img_mask_copy = np.ones(img_mask.shape) * img_mask * 255
            img_mask_copy = img_mask_copy.astype(np.uint8)
            cv.imshow('origin_mask', img_mask_copy)


            # Perform initial dilation
            img, img_mask = self.morphology(img, img_mask, 'ini_dilate')

            img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
            cv.imshow('initial', img)
            img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            img_mask_copy = np.ones(img_mask.shape) * img_mask * 255
            img_mask_copy = img_mask_copy.astype(np.uint8)
            cv.imshow('initial_mask', img_mask_copy)


            # do small hole closure
            img, img_mask = self.morphology(img, img_mask, 'close')

            img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
            cv.imshow('closure', img)
            img_mask_copy = np.ones(img_mask.shape) * img_mask * 255
            img_mask_copy = img_mask_copy.astype(np.uint8)
            cv.imshow('mask', img_mask_copy)

            cv.waitKey(0)
            img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

            # small hole fill
            img, img_mask = self.morphology(img, img_mask, 'hole_fill')

            img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
            cv.imshow('hole_fill', img)
            img_mask_copy = np.ones(img_mask.shape) * img_mask * 255
            img_mask_copy = img_mask_copy.astype(np.uint8)
            cv.imshow('mask', img_mask_copy)
            cv.waitKey(0)
            img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

            # if unknown pixels still exist:
            if len(np.unique(img_mask)) > 1:
                land_mask = fetch_land_mask()
                img_copy = img * ~land_mask
                img_copy = inpaint.inpaint_biharmonic(img_copy, img_mask, channel_axis=-1)
                img = img * land_mask + img_copy * ~land_mask

            img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
            cv.imshow('res', img)
            cv.waitKey(0)
            exit()

            path = os.path.join(self.save_path, self.img_paths[count][-8:-4] + '_res.tif')
            cv.imwrite(path, img)
            count += 1
        print('inpainting completed')


if __name__ == '__main__':
    model = InpaintModel(img_paths=('../data/dataset/train/0718.tif', ), save_path=" ")
    model.inpaint()
