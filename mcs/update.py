import numpy as np
import math

from . import global_val as gv


def updtrec(j, s, f):
    if len(gv.record) < s:
        gv.record[s, 0] = j
    elif gv.record[s, 0] == 0:
        gv.record[s, 0] = j
    elif f[j] < f[gv.record[s, 0]]:
        gv.record[s, 0] = j


def updtoptl(i, x, y, iopt, level, f):

    for j in range(0, len(iopt)):
        if min(x, y) <= gv.xglob[i, iopt[j]]:
            if gv.xglob[i, iopt[j]] <= max(x, y):
                gv.optlevel[iopt[j]] = level
                gv.foptbox[iopt[j]] = f;
