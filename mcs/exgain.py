import numpy as np
import math

from .polint import polint
from .quadops import quadmin, quadpol
from .subint import subint


def exgain(n, n0, l, L, x, y, x1, x2, fx, f0, f1, f2):
    e = np.zeros((n))
    emin = math.inf

    isplit = 0
    splval = 0

    for i in range(n):
        if n0[i] == 0:
            # expected gain for splitting according
            # to the initialization list
            e[i] = min(f0[0:L[i, 0]-1, i]) - f0[l[i, 0]-1, i]
            if e[i] < emin:
                emin = e[i]
                isplit = i + 1
                splval = math.inf
        else:
            z1 = [x[i], x1[i], x2[i]]
            z2 = [0, f1[i] - fx, f2[i] - fx]
            d = polint(z1, z2)
            [eta1, eta2] = subint(x[i], y[i])
            # safeguard against splitting too close
            # to x[i]
            xi1 = min(eta1, eta2)
            xi2 = max(eta1, eta2)
            z = quadmin(xi1, xi2, d, z1)
            e[i] = quadpol(z, d, z1)
            if e[i] < emin:
                emin = e[i]
                isplit = i + 1
                splval = z

    return e, isplit, splval
