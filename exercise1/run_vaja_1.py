import numpy as np
import matplotlib.pyplot as plt
from ex1_utils import rotate_image, show_flow
from of_methods import lucas_kanade, horn_schunck
import cv2
from time import time


im1 = np.random.rand(200, 200).astype(np.float32)
im2 = im1.copy()
im2 = rotate_image(im2, -1)


path = ".\\exercise1\\lab2\\"
im1 = cv2.imread(path + "010.jpg", 0)
im2 = cv2.imread(path + "011.jpg", 0)

start = time()
U_lk, V_lk = lucas_kanade(im1, im2, 3)
print(time() - start)
fig1, ((ax1_11, ax1_12), (ax1_21, ax1_22)) = plt.subplots(2, 2)
ax1_11.imshow(im1)
ax1_12.imshow(im2)
show_flow(U_lk, V_lk, ax1_21, type="angle")
show_flow(U_lk, V_lk, ax1_22, type="field", set_aspect=True)
fig1.suptitle("Lucas−Kanade Optical Flow")

start = time()
U_hs, V_hs = horn_schunck(im1, im2, 100, 0.5)
print(time() - start)
fig2, ((ax2_11, ax2_12), (ax2_21, ax2_22)) = plt.subplots(2, 2)
ax2_11.imshow(im1)
ax2_12.imshow(im2)
show_flow(U_hs, V_hs, ax2_21, type="angle")
show_flow(U_hs, V_hs, ax2_22, type="field", set_aspect=True)
fig2.suptitle("Horn−Schunck Optical Flow")

"""
U_1, V_1 = lucas_kanade(im1, im2, 3)
U_2, V_2 = lucas_kanade(im1, im2, 9)
fig1, ((ax1_11, ax1_12), (ax1_21, ax1_22)) = plt.subplots(2, 2)
ax1_11.imshow(im1)
ax1_12.imshow(im2)
show_flow(U_1, V_1, ax1_21, type="angle")
show_flow(U_1, V_1, ax1_22, type="field", set_aspect=True)
fig1.suptitle("Lucas−Kanade Optical Flow n=3")
fig2, ((ax2_11, ax2_12), (ax2_21, ax2_22)) = plt.subplots(2, 2)
ax2_11.imshow(im1)
ax2_12.imshow(im2)
show_flow(U_2, V_2, ax2_21, type="angle")
show_flow(U_2, V_2, ax2_22, type="field", set_aspect=True)
fig2.suptitle("Lucas−Kanade Optical Flow n=9")

U_1, V_1 = horn_schunck(im1, im2, 100, 0.5)
U_2, V_2 = horn_schunck(im1, im2, 1000, 0.5)
fig1, ((ax1_11, ax1_12), (ax1_21, ax1_22)) = plt.subplots(2, 2)
ax1_11.imshow(im1)
ax1_12.imshow(im2)
show_flow(U_1, V_1, ax1_21, type="angle")
show_flow(U_1, V_1, ax1_22, type="field", set_aspect=True)
fig1.suptitle("Horn-Schunck Optical Flow n_iter=100")
fig2, ((ax2_11, ax2_12), (ax2_21, ax2_22)) = plt.subplots(2, 2)
ax2_11.imshow(im1)
ax2_12.imshow(im2)
show_flow(U_2, V_2, ax2_21, type="angle")
show_flow(U_2, V_2, ax2_22, type="field", set_aspect=True)
fig2.suptitle("Horn-Schunck Optical Flow n_iter=1000")
"""

plt.show()
