import numpy as np
from numpy import linalg
from abc import abstractmethod
import pandas as pd
import math

pd.options.display.float_format = '{:,.10f}'.format
np.set_printoptions(suppress=True, precision=10)

TOR = pow(10.0, -5)


class NewtonMethod(object):

    def __init__(self):
        return

    @abstractmethod
    def f(self, x):
        return NotImplementedError('Implement f()!')

    @abstractmethod
    def jacobian(self, x):
        return NotImplementedError('Implement jacobian()!')

    @abstractmethod
    def run(self, x):
        return NotImplementedError('Implement run()!')


class Newton(NewtonMethod):

    def __init__(self):
        super(NewtonMethod, self).__init__()

    def f(self, x):
        sol = np.zeros(len(x))
        sol[0] = 3 * x[0] - math.cos(x[1] * x[2]) - 1.0 / 2.0
        sol[1] = pow(x[0], 2) - 81 * pow(x[1] + 0.1, 2) + math.sin(x[2]) + 1.06
        sol[2] = math.exp(-x[0] * x[1]) + 20 * x[2] + (10 * math.pi - 3.0) / 3.0
        return sol

    def jacobian(self, x):
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
            jac = self.jacobian(x)
            f = -self.f(x)
            y = linalg.solve(jac, f)
            nx = x + y
            residual = linalg.norm(x - nx, np.inf)
            x = nx
            
            row = len(df)
            df.loc[row] = [nxe for nxe in nx] + [residual, np.nan]
            if residual < TOR:
                break

        for i in range(len(df)):
            xk = np.array([df.loc[i][j] for j in range(len(x))])
            df.loc[i][4] = linalg.norm(x - xk, np.inf)

        print(df)


def main():
    x0 = np.array([0.1, 0.1, -0.1])
    Newton().run(x0)
    return


if __name__ == '__main__':
    main()
