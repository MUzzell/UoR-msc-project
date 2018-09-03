import numpy as np
import math

from .util import find

def neighbour(x, delta, u, v):
    i1 = find(x, lambda a: a == u)
    i2 = find(x, lambda a: a == v)
    x1 = max(u, x-delta)
    x2 = min(x+delta, v)

    x1[i1] = x[i1] + 2 * delta[i1]
    x2[i2] = x[i2] - 2 * delta[i2]

    return i1, i2