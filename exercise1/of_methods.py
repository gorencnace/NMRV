from random import gauss
from ex1_utils import gaussderiv, rotate_image, show_flow, gausssmooth
import numpy as np
import cv2
from scipy.signal import convolve2d


def lucas_kanade(im1, im2, N):
    threshold = 0.5
    Ix1, Iy1 = gaussderiv(im1, 1)
    Ix2, Iy2 = gaussderiv(im2, 1)
    Ix = (Ix1 + Ix2) / 2
    Iy = (Iy1 + Iy2) / 2
    It = gausssmooth(im2 - im1, 1)

    kernel = np.ones((N, N))

    IxIx = convolve2d(Ix ** 2, kernel, "same")
    IyIy = convolve2d(Iy ** 2, kernel, "same")
    IxIy = convolve2d(Ix * Iy, kernel, "same")
    IxIt = convolve2d(Ix * It, kernel, "same")
    IyIt = convolve2d(Iy * It, kernel, "same")

    D = IxIx * IyIy - IxIy ** 2
    T = IxIx + IyIy
    k = 0.05
    r = D - k * T
    r[r >= threshold] = 1
    r[r < threshold] = 0

    D[D == 0] = 1000

    U = (IxIy * IyIt - IyIy * IxIt) * r / D
    V = (IxIy * IxIt - IxIx * IyIt) * r / D

    return U, V


def horn_schunck(im1, im2, n_iters, lmbd):
    im1 = gausssmooth(im1 / 255, 1)
    im2 = gausssmooth(im2 / 255, 1)

    x_kernel = np.array([[-1, 1], [-1, 1]]) * 0.5
    y_kernel = np.array([[-1, -1], [1, 1]]) * 0.5
    t_kernel = np.ones((2, 2)) * 0.25
    Ld = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) * 0.25

    u, v = np.zeros_like(im1), np.zeros_like(im1)

    Ix = convolve2d(im1, x_kernel, "same") + convolve2d(im2, x_kernel, "same")
    # cv2.imshow("", Ix)
    # cv2.waitKey(0)

    Iy = convolve2d(im1, y_kernel, "same") + convolve2d(im2, y_kernel, "same")
    # cv2.imshow("", Iy)
    # cv2.waitKey(0)

    It = convolve2d(im2, t_kernel, "same") - convolve2d(im1, t_kernel, "same")
    # cv2.imshow("", It)
    # cv2.waitKey(0)

    D = lmbd + Ix ** 2 + Iy ** 2

    for _ in range(n_iters):
        ua = convolve2d(u, Ld, "same")
        va = convolve2d(v, Ld, "same")

        P = Ix * ua + Iy * va + It

        pd = P / D
        u = ua - Ix * pd
        v = va - Iy * pd

    return u, v
