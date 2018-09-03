import numpy as np
import math

def polint(x, f):
    d = np.zeros((3), dtype=np.float64)

    d[0] = f[0]
    d[1] = (f[1] - f[0])/(x[1] - x[0])
    f23  = (f[2] - f[1])/(x[2] - x[1])
    d[2] = (f23 - d[1])/(x[2] - x[0])

    return d