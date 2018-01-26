import numpy as np
from numpy import linalg

import pandas as pd
import math
np.set_printoptions(suppress=True)

TOR = pow(10.0, -5)


def fix_point(x):
    """
    given x_0 in R^3 as a starting point.

    :param x: x_0 as described
    :return: the minimizer x* of f
    """
    y = np.zeros(len(x))
    while True:
        y[0] = math.cos(x[1] * x[2]) / 3.0 + 1.0 / 6.0
        y[1] = math.sqrt(x[0] * x[0] + math.sin(x[2]) + 1.06) / 9.0 - 0.1
        y[2] = -math.exp(-x[0] * x[1]) / 20.0 - (10.0 * math.pi - 3.0) / 60.0
        if linalg.norm(x - y, np.inf) < TOR:
            break
        x = y

    print(y)


def fix_point_acc(x):
    y = np.zeros(len(x))
    while True:
        y[0] = math.cos(x[1] * x[2]) / 3.0 + 1.0 / 6.0
        y[1] = math.sqrt(y[0] * y[0] + math.sin(x[2]) + 1.06) / 9.0 - 0.1
        y[2] = -math.exp(-y[0] * y[1]) / 20.0 - (10.0 * math.pi - 3.0) / 60.0
        if linalg.norm(x - y, np.inf) < TOR:
            break
        x = y

    print(y)


def main():
    x0 = np.array([0.1, 0.1, -0.1])
    fix_point(x0)
    fix_point_acc(x0)
    return


if __name__ == '__main__':
    main()
