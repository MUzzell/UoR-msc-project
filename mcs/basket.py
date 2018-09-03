import numpy as np
import math

from . import global_val as gv

def basket(fun, data, x, f, xmin, fmi, xbest,
           fbest, stop, nbasket):
    loc = 1
    flag = 1
    ncall = 0

    if not nbasket:
        return xbest, fbest, xmin, fmi, x, f, loc, flag, ncall

    for k in range(nbasket):
        pass