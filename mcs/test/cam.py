import numpy as np


def cam(data, x=None):
    x1 = 0; x2 = 0

    if data is None or not data:
        x1 = x[0]; x2 = x[1]
    else:
        x1 = data; x2 = x

    # (4-2.1.*x1.^2+x1.^4./3).*x1.^2+x1.*x2+(-4+4.*x2.^2).*x2.^2
    r1 = np.multiply(2.1, np.power(x1, 2))
    r1 = 4 - r1 + np.divide(np.power(x1, 4), 3)
    r1 = np.multiply(r1, np.power(x1, 2))
    r2 = np.multiply(x1, x2)
    r3 = -4 + np.multiply(4, np.power(x2, 2))
    r3 = np.multiply(r3, np.power(x2, 2))

    return r1 + r2 + r3
