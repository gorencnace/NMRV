from random import gauss
from ex1_utils import gaussderiv, rotate_image, show_flow, gausssmooth
import numpy as np
from scipy.signal import convolve2d
import cv2


def lucas_kanade(im1, im2, N):
    Ix1, Iy1 = gaussderiv(im1, 1)
    Ix2, Iy2 = gaussderiv(im2, 1)
    Ix = gausssmooth((Ix1 + Ix2) / 2, 1)
    Iy = gausssmooth((Iy1 + Iy2) / 2, 1)
    It = gausssmooth(im2 - im1, 1)

    kernel = np.ones((N, N))

    IxIx = convolve2d(Ix * Ix, kernel, "same")
    IyIy = convolve2d(Iy * Iy, kernel, "same")
    IxIy = convolve2d(Ix * Iy, kernel, "same")
    IxIt = convolve2d(Ix * It, kernel, "same")
    IyIt = convolve2d(Iy * It, kernel, "same")

    D = IxIx * IyIy - IxIy ** 2 + 0.0001
    U = (IxIy * IyIt - IyIy * IxIt) / D
    V = (IxIy * IxIt - IxIx * IyIt) / D
    return U, V


def horn_schunck(im1, im2, n_iters, lmbd):
    Ld = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 4
    u, v = np.zeros_like(im1), np.zeros_like(im1)
    Ix1, Iy1 = gaussderiv(im1, 1)
    Ix2, Iy2 = gaussderiv(im2, 1)
    Ix = gausssmooth((Ix1 + Ix2) / 2, 1)
    Iy = gausssmooth((Iy1 + Iy2) / 2, 1)
    It = gausssmooth(im2 - im1, 1)

    D = lmbd + Ix * Ix + Iy * Iy

    for _ in range(n_iters):
        ua = convolve2d(u, Ld, "same")
        va = convolve2d(v, Ld, "same")

        P = Ix * ua + Iy * va + It

        pd = P / D
        u = ua - Ix * pd
        v = va - Iy * pd

    return u, v
