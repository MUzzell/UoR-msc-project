import numpy as np
import math
import pdb

from . import global_val as gv


def strtsw(smax, level, f):
    '''
    updates the record list for starting a new sweep and computes the
    lowest level containing non-split boxes

    Input:
    smax             depth of search
    level(0:nboxes)  levels of these boxes (level(j) = 0 if box j has
                     already been split)
    f(0:nboxes)      their function values at the base vertices

    Output:
    s            lowest level with record(s) ~= 0
    '''

    gv.record = np.zeros((smax-1, 1), dtype=int)
    s = smax

    for j in range(0, gv.nboxes):
        if level[j] > 0:
            if level[j] < s:
                s = level[j]

            if not gv.record[level[j]-1]:
                gv.record[level[j]-1] = j
            elif f[j] < f[gv.record[level[j]-1]]:
                gv.record[level[j]-1] = j

    return s
