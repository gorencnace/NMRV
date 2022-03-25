import numpy as np
from ex2_utils import generate_responses_1, get_patch

res = generate_responses_1()

l = 3
xi, yi = np.meshgrid(np.arange(-l, l + 1), np.arange(-l, l + 1))

# starting position
x, y = 60, 50

condition = True
n_iter = 500
i = 1
while condition:
    wi, inliers = get_patch(res, [x, y], (7, 7))
    wi_sum = np.sum(wi)
    xk = np.sum(xi * wi) / wi_sum
    yk = np.sum(yi * wi) / wi_sum

    x += xk
    y += yk

    i += 1

    if xk < 1e-4 and yk < 1e-4 or i > n_iter:
        condition = False

print(f"x={int(x)}, y={int(y)}")
print(i)
print(res[int(x), int(y)])
print(np.max(res))
