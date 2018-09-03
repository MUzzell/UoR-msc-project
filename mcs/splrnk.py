import numpy as np
import math

from .split import split2


def splrnk(n, n0, p, x, y):
    """
    determines the splitting index and splitting value for splitting a
    box by rank
    """
    isplit = 1
    n1 = n0[0, 0]
    p1 = p[0]

    for i in range(1, n):
        if n0[i, 0] < n1 or (n0[i, 0] == n1 and p[i] < p1):
            isplit = i + 1
            n1 = n0[i]
            p1 = p[i]

    if n1 > 0:
        splval = split2(x[isplit-1, 0], y[isplit-1, 0])
    else:
        splval = math.inf

    return isplit, splval
