from ex2_utils import (
    Tracker,
    extract_histogram,
    backproject_histogram,
    create_epanechnik_kernel,
)
import numpy as np


class MSTracker(Tracker):
    def initialize(self, image, region):

        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [
                np.min(x_),
                np.min(y_),
                np.max(x_) - np.min(x_) + 1,
                np.max(y_) - np.min(y_) + 1,
            ]

        # width and height must be odd
        if region[2] % 2 == 0:
            region[2] -= 1
        if region[3] % 2 == 0:
            region[3] -= 1

        left = max(region[0], 0)
        top = max(region[1], 0)

        right = min(region[0] + region[2], image.shape[1] - 1)
        bottom = min(region[1] + region[3], image.shape[0] - 1)

        self.template = image[int(top) : int(bottom), int(left) : int(right)]
        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)
        self.size = (region[2], region[3])
        self.weights = create_epanechnik_kernel(region[2], region[3], 4)

        self.histogram_q = extract_histogram(
            self.template, self.parameters.nbins, self.weights
        )

    def track(self, image):
        x, y = self.position
        condition = True
        n_iter = 500
        i = 1
        while condition:
            left = max(round(x - self.size[0] / 2), 0)
            top = max(round(y - self.size[1] / 2), 0)
            right = min(round(x + self.size[0] / 2), image.shape[1] - 1)
            bottom = min(round(y + self.size[1] / 2), image.shape[0] - 1)
            template = image[int(top) : int(bottom), int(left) : int(right)]

            if i > 1:
                histogram_q = extract_histogram(
                    template, self.parameters.nbins, self.weights
                )
                self.histogram_q = (
                    1 - self.parameters.alpha
                ) * self.histogram_q + self.parameters.alpha * histogram_q

            histogram_p = extract_histogram(
                template, self.parameters.nbins, self.weights
            )

            v = np.sqrt(self.histogram_q / (histogram_p + self.parameters.eps))

            wi = backproject_histogram(template, v, self.parameters.nbins)
            l = int(np.floor(np.shape(wi)[0] / 2))
            w = int(np.floor(np.shape(wi)[1] / 2))
            xi, yi = np.meshgrid(np.arange(-w, w + 1), np.arange(-l, l + 1))
            wi_sum = np.sum(wi)
            xk = np.sum(xi * wi) / wi_sum
            yk = np.sum(yi * wi) / wi_sum
            x += xk
            y += yk
            i += 1

            if xk < 1e-3 and yk < 1e-3 or i > n_iter:
                condition = False
        return [x - self.size[0] / 2, y - self.size[1] / 2, self.size[0], self.size[1]]


class MSParams:
    def __init__(self):
        self.nbins = 16
        self.alpha = 0.5
        self.eps = 1e-4
