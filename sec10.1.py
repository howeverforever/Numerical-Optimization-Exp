import numpy as np
from numpy import linalg
from abc import abstractmethod
import pandas as pd
import math

pd.options.display.float_format = '{:,.8f}'.format
np.set_printoptions(suppress=True, precision=8)

TOR = pow(10.0, -5)
MAX_ITR = 30


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


class FixedPoint1D(FixedPointMethod):

    def __init__(self):
        super(FixedPointMethod, self).__init__()

    def f(self, x):
        return math.cos(x)

    def g1(self, x):
        return np.array(x - pow(x, 3) - 4 * pow(x, 2) + 10)

    def g2(self, x):
        return np.array(math.sqrt(10 / x - 4 * x))

    def g3(self, x):
        return np.array(math.sqrt(10 - pow(x, 3)) / 2)

    def g4(self, x):
        return np.array(math.sqrt(10 / (4 + x)))

    def g5(self, x):
        return np.array(x - (pow(x, 3) + 4 * pow(x, 2) - 10) / (3 * pow(x, 2) + 8 * x))

    def run2(self, x0):
        df = pd.DataFrame(columns=['(sp) f(x)'])
        row = len(df)
        x = x0
        df.loc[row] = [x]
        for k in range(MAX_ITR):
            try:
                y = self.f(x)
            except ValueError:
                break
            residual = math.fabs(x - y)
            x = y

            row = len(df)
            df.loc[row] = [y]
            if residual < TOR or x > 1e9:
                break
        return df

    def run(self, x0):
        """
        given x_0 in R^3 as a starting point.

        :param x: x_0 as described
        :return: the minimizer x* of f
        """
        g = [self.g1, self.g2, self.g3, self.g4, self.g5]

        total_df = None
        for j in range(len(g)):
            df = pd.DataFrame(columns=['g' + str(j + 1) + '(x)'])
            row = len(df)
            x = x0
            df.loc[row] = [x]
            for k in range(MAX_ITR):
                try:
                    y = np.array(g[j](x))
                except ValueError:
                    break
                residual = math.fabs(x - y)
                x = y

                row = len(df)
                df.loc[row] = [y]
                if residual < TOR or x > 1e9:
                    break
            total_df = df if total_df is None else pd.concat([total_df, df], axis=1)


def main():
    x0 = np.array([0.1, 0.1, -0.1])
    x10 = 1.5
    # FixedPoint().run(x0)
    # FixedPointAcceleration().run(x0)
    # FixedPoint1D().run(x10)
    FixedPoint1D().run2(math.pi / 4)

    return


if __name__ == '__main__':
    main()
