import numpy as np
from numpy import linalg
from abc import abstractmethod
import pandas as pd
import math

pd.options.display.float_format = '{:,.6f}'.format
np.set_printoptions(suppress=True, precision=6)

TOR = pow(10.0, -5)
TRUE_X = np.array([0.5, 0, -0.5235988])


class SteepestDescentMethod(object):

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


class SteepestDescent(SteepestDescentMethod):

    def __init__(self):
        super(SteepestDescentMethod, self).__init__()

    def f(self, x):
        sol = np.zeros(len(x))
        sol[0] = 3 * x[0] - math.cos(x[1] * x[2]) - 1.0 / 2.0
        sol[1] = pow(x[0], 2) - 81 * pow(x[1] + 0.1, 2) + math.sin(x[2]) + 1.06
        sol[2] = math.exp(-x[0] * x[1]) + 20 * x[2] + (10 * math.pi - 3.0) / 3.0
        return sol

    def g(self, x):
        sol = self.f(x)
        return sum([e * e for e in sol])

    def grad_g(self, x):
        return 2 * self.jacobian(x).transpose().dot(self.f(x))

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
        df = pd.DataFrame(columns=['x' + str(i + 1) for i in range(len(x))] + ['g', 'residual', 'actual-residual'])

        row = len(df)
        df.loc[row] = [xe for xe in x] + [self.g(x), np.nan, np.nan]

        while True:
            prev_x = x
            g1 = self.g(x)
            z = self.grad_g(x)
            z0 = linalg.norm(z, 2)
            if z0 == 0.0:
                print('Zero gradient')
                return x

            z /= z0
            alpha3 = 1
            g3 = self.g(x - alpha3 * z)
            while g3 >= g1:
                alpha3 /= 2.0
                g3 = self.g(x - alpha3 * z)
                if alpha3 < TOR / 2.0:
                    print('No likely improvement')
                    return x

            alpha2 = alpha3 / 2.0
            g2 = self.g(x - alpha2 * z)

            h1 = (g2 - g1) / alpha2
            h2 = (g3 - g2) / (alpha3 - alpha2)
            h3 = (h2 - h1) / alpha3

            alpha0 = (alpha2 - h1 / h3) / 2.0
            g0 = self.g(x - alpha0 * z)

            alpha = alpha0
            g = g0
            if g3 < g:
                alpha = alpha3
                g = g3

            x = x - alpha * z
            residual = linalg.norm(x - prev_x, np.inf)
            row = len(df)
            df.loc[row] = [nxe for nxe in x] + [g, residual, np.nan]
            if math.fabs(g - g1) < TOR:
                break

        for i in range(len(df)):
            xk = np.array([df.loc[i][j] for j in range(len(x))])
            df.loc[i][5] = linalg.norm(xk - x, np.inf)

        print(df)


def main():
    x0 = np.array([0, 0, 0])
    SteepestDescent().run(x0)


if __name__ == '__main__':
    main()
