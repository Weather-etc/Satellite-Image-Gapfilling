"""
This file provides several ways to evaluate the similarity of different images.
1.MSE: Mean squared error. Refer to Wiki(https://en.wikipedia.org/wiki/Mean_squared_error)
2.RMSE: Rooted mean squared error. Refer to Wiki(https://en.wikipedia.org/wiki/Root-mean-square_deviation)
"""
import numpy as np


def mse(img0, img1):
    """
    This function computes MSE of image img0 and image img1.
    :param img0: a numpy.uint8 numpy.ndarray.
    :param img1: a numpy.uint8 numpy.ndarray.
    :return: float.
    """
