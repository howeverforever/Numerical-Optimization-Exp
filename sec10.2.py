import numpy as np
from numpy import linalg

import pandas as pd
import math

np.set_printoptions(suppress=True, precision=12)

TOR = pow(10.0, -5)


def f(x):
    sol = np.zeros(len(x))
    sol[0] = 3 * x[0] - math.cos(x[1] * x[2]) - 1.0 / 2.0
    sol[1] = pow(x[0], 2) - 81 * pow(x[1] + 0.1, 2) + math.sin(x[2]) + 1.06
    sol[2] = math.exp(-x[0] * x[1]) + 20 * x[2] + (10 * math.pi - 3.0) / 3.0
    return sol


def jacobian(x):
    jac = np.zeros(shape=(3, 3))
    jac[0][0] = 3.0
    jac[0][1] = x[2] * math.sin(x[1] * x[2])
    jac[0][2] = x[1] * math.sin(x[1] * x[2])
    jac[1][0] = 2 * x[0]
    jac[1][1] = -162 * (x[1] + 0.1)
    jac[1][2] = math.cos(x[2])
    jac[2][0] = -x[1] * math.exp(-x[0] * x[1])
    jac[2][1] = -x[0] * math.exp(-x[0] * x[1])
    jac[2][2] = 20
    return jac


def newton_method(x):
    """
    given x_0 in R^3 as a starting point.

    :param x: x_0 as described
    :return: the minimizer x* of f
    """
    while True:
        print(x)
        jac = jacobian(x)
        ff = -f(x)
        y = linalg.solve(jac, ff)
        nx = x + y
        if linalg.norm(x - nx, np.inf) < TOR:
            break
        x = nx
    return x


def main():
    x0 = np.array([0.1, 0.1, -0.1])
    newton_method(x0)
    return


if __name__ == '__main__':
    main()
