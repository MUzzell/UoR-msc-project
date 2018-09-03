import numpy as np
import math

def quadpol(x, d, x0):

    m1 = np.multiply(d[1], x - x0[0])
    m2 = np.multiply(d[2], x - x0[0])
    m2 = np.multiply(m2,   x - x0[1])

    return d[0] + m1 + m2

def quadmin(a, b, d, x0):
    x = 0
    if d[2] == 0:
        if d[1] > 0:
            x = a
        else:
            x = b
    elif d[2] > 0:
        x1 = np.multiply(0.5, x0[0] + x0[1])
        x1 = x1 - np.multiply(0.5, np.divide(d[1], d[2]))

        if a <= x1 and x1 <= b:
            x = x1
        elif quadpol(a,d,x0) < quadpol(b,d,x0):
            x = a
        else:
            x = b
    else:
        if quadpol(a,d,x0) < quadpol(b,d,x0):
            a = a
        else:
            x = b

    return x