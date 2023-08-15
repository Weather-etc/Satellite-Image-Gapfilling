import os
import sys
import cv2 as cv
import numpy as np
from tools.mask import get_mask, fetch_land_mask
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
        if kernel.ndim != img.ndim:
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

        indices = np.where(mask == 0)
        indices = zip(*indices)
        shape = kernel.shape
        output = np.zeros(img.shape)

        img = np.pad(img, (shape[0] // 2, shape[1] // 2), 0)
        for index in indices:
            range0 = np.array(index - np.array(shape) / 2)
            item_mask = range0 > 0
            range0 = range0 * item_mask + np.zeros(range0.shape) * ~item_mask
            range0 = range0.astype(np.uint8)
            range1 = np.array(index + np.array(shape) / 2)
            item_mask = range1 > 0
            range1 = range1 * item_mask + np.zeros(range1.shape) * ~item_mask
            range1 = range1.astype(np.uint8)

            print(range0)
            print(range1)

            submat = img[range0[0]:range1[0], range0[1]:range1[1]]
            print(submat.shape)
            print(kernel.shape)
            print('------')
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
        new_mask = img_mask.astype(np.uint8)
        if opt == 'ini_dilate':
            iters = 1
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 3))
            morph_v = cv.dilate(morph_v, kernel, iterations=iters)
            morph_hs = cv.dilate(morph_hs, kernel, iterations=iters)
            new_mask = ~cv.dilate(~new_mask, kernel, iterations=iters)
        elif opt == 'close':
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
            morph_v = cv.morphologyEx(morph_v, cv.MORPH_CLOSE, kernel)
            morph_hs = cv.morphologyEx(morph_hs, cv.MORPH_CLOSE, kernel)
            new_mask = ~cv.morphologyEx(~new_mask, cv.MORPH_CLOSE, kernel)
        elif opt == 'hole_fill':
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
            morph_v = self.dilate_masked(morph_v, img_mask, kernel)
            morph_hs = self.dilate_masked(morph_hs, img_mask, kernel)
            new_mask = ~cv.morphologyEx(~new_mask, img_mask, kernel)
        else:
            print(Fore.RED + 'ERROR: Unsupported morphological operation: ' + opt)
            exit(1)
        # generate result
        res = np.zeros(img.shape)
        new_mask = new_mask.astype(bool)
        res[..., 0] = morph_hs[..., 0] * ~land_mask + img_hs[..., 0] * land_mask
        res[..., 1] = morph_hs[..., 1] * ~land_mask + img_hs[..., 1] * land_mask
        res[..., 2] = morph_v * ~land_mask + img_v * land_mask
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
            # Perform initial dilation
            img, img_mask = self.morphology(img, img_mask, 'ini_dilate')
            # do small hole closure
            img, img_mask = self.morphology(img, img_mask, 'close')
            # small hole fill
            img, img_mask = self.morphology(img, img_mask, 'hole_fill')

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
