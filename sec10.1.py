import numpy as np
from numpy import linalg
from abc import abstractmethod
import pandas as pd
import math

pd.options.display.float_format = '{:,.8f}'.format
np.set_printoptions(suppress=True, precision=8)

TOR = pow(10.0, -5)


class FixedPointMethod(object):

    def __init__(self):
        return

    @abstractmethod
    def f(self, x):
        return NotImplementedError('Implement f()!')

    @abstractmethod
    def run(self, x):
        return NotImplementedError('Implement run()!')


class FixedPoint(FixedPointMethod):

    def __init__(self):
        super(FixedPointMethod, self).__init__()

    def f(self, x):
        sol = np.zeros(len(x))
        sol[0] = math.cos(x[1] * x[2]) / 3.0 + 1.0 / 6.0
        sol[1] = math.sqrt(x[0] * x[0] + math.sin(x[2]) + 1.06) / 9.0 - 0.1
        sol[2] = -math.exp(-x[0] * x[1]) / 20.0 - (10.0 * math.pi - 3.0) / 60.0
        return sol

    def run(self, x):
        """
        given x_0 in R^3 as a starting point.

        :param x: x_0 as described
        :return: the minimizer x* of f
        """
        df = pd.DataFrame(columns=['x' + str(i + 1) for i in range(len(x))] + ['residual', 'actual-residual'])

        row = len(df)
        df.loc[row] = [xe for xe in x] + [np.nan, np.nan]

        while True:
            y = self.f(x)
            residual = linalg.norm(x - y, np.inf)
            x = y

            row = len(df)
            df.loc[row] = [ye for ye in y] + [residual, np.nan]
            if residual < TOR:
                break

        for i in range(len(df)):
            xk = np.array([df.loc[i][j] for j in range(len(x))])
            df.loc[i][4] = linalg.norm(x - xk, np.inf)

        print(df)


class FixedPointAcceleration(FixedPointMethod):

    def __init__(self):
        super(FixedPointMethod, self).__init__()

    def f(self, x):
        sol = np.zeros(len(x))
        sol[0] = math.cos(x[1] * x[2]) / 3.0 + 1.0 / 6.0
        sol[1] = math.sqrt(sol[0] * sol[0] + math.sin(x[2]) + 1.06) / 9.0 - 0.1
        sol[2] = -math.exp(-sol[0] * sol[1]) / 20.0 - (10.0 * math.pi - 3.0) / 60.0
        return sol

    def run(self, x):
        """
        given x_0 in R^3 as a starting point.

        :param x: x_0 as described
        :return: the minimizer x* of f
        """
        df = pd.DataFrame(columns=['x' + str(i + 1) for i in range(len(x))] + ['residual', 'actual-residual'])

        row = len(df)
        df.loc[row] = [xe for xe in x] + [np.nan, np.nan]
        while True:
            y = self.f(x)
            residual = linalg.norm(x - y, np.inf)
            x = y

            row = len(df)
            df.loc[row] = [ye for ye in y] + [residual, np.nan]
            if residual < TOR:
                break

        for i in range(len(df)):
            xk = np.array([df.loc[i][j] for j in range(len(x))])
            df.loc[i][4] = linalg.norm(x - xk, np.inf)

        print(df)


def main():
    x0 = np.array([0.1, 0.1, -0.1])
    FixedPoint().run(x0)
    FixedPointAcceleration().run(x0)
    return


if __name__ == '__main__':
    main()
