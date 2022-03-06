from exercise1.ex1_utils import gaussderiv, rotate_image, show_flow
import numpy as np
from scipy.signal import convolve2d
import cv2

def lucas_kanade(im1, im2, N):
    Ix, Iy = gaussderiv(im1, 0.75)
    It = im2 - im1

    kernel = np.ones((N, N))

    Ix2 = convolve2d(Ix * Ix, kernel, 'same')
    Iy2 = convolve2d(Iy * Iy, kernel, 'same')
    IxIy = convolve2d(Ix * Iy, kernel, 'same')
    IxIt = convolve2d(Ix * It, kernel, 'same')
    IyIt = convolve2d(Iy * It, kernel, 'same')

    D = Ix2 * Iy2 - IxIy ** 2

    U = (IxIy * IyIt - Iy2 * IxIt) / D
    V = (IxIy * IxIt - Ix2 * IyIt) / D

    return U, V


def horn_schunck(im1, im2, n_iters, lmbd):
    pass


if __name__ == '__main__':
    path = '.\\exercise1\\lab2\\'
    img1 = cv2.imread(path + '001.jpg', 0)
    img2 = rotate_image(img1, 5)
    u, v = lucas_kanade(img1, img2, 3)
    show_flow(u, v)