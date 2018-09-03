import numpy as np
import math

from .check import chrelerr
from .util import genbox, max_idx
from .split import split1
from .update import updtrec, updtoptl
from . import global_val as gv


def splinit(fun, data, i, s, smax, par, x0, n0, u, v, x, y, x1, x2,
            L, l, xmin, fmi, ipar, level, ichild, f, xbest, fbest, stop):
    """
    splits box # par at level s according to the initialization list
    in the ith coordinate and inserts its children and their parameters
    in the list
    """
    ncall = 0
    n = len(x)
    f0 = np.zeros((max(L), 1))
    flag = 1

    # if prt > 1

    for j in range(L[i]):
        if j != l(i):
            x[i] = x0[i, j]
            f0[j] = fun(data, x)
            ncall = ncall + 1
            if f0[j] < fbest:
                fbest = f0[j]
                xbest = x
                gv.nsweepbest = nsweepbest
                if stop[0] > 0 and stop[0] < 1:
                    flag = chrelerr(fbest, stop)
                elif stop[0] == 0:
                    flag = chvtr(fbest, stop[1])

                if !flag:
                    return (xbest, fbest, f0, xmin, fmi, ipar, level,
                            ichild, f, flag, ncall)

        else:
            f0[j] = f[0, par]

    fm, i1 = max_idx(f0)
    if i1 > 1:
        splval1 = split1(x0[i, i1 - 1], x0[i, i1], f0[i1 - 1], f0[i1])
    else:
        splval = u[i]

    if i1 < L[i]:
        splval2 = split1(x0[i, i1], x0[i, i1 + 1], f0[i1], f0[i1 + 1])
    else:
        splval2 = v[i]

    if s + 1 < smax:
        nchild = 0
        if u[i] < x0[i, 0]:
            nchild = nchild + 1
            nboxes = nboxes + 1
            ipar, level, ichild, f = genbox(
                ipar, level, ichild, f,
                par, s + 1, -nchild, f0[0])
            updtrec(gv.nboxes, level[gv.nboxes, f[0, :]])
            # if prt > 1

        for j in range(L[i] - 1):
            nchild = nchild + 1
            splval = split1(x0[i, j], x0[i, j+1], f0[j], f0[j+1])
            if f0[j] <= f0[j+1] or s + 2 < smax:
                gv.nboxes = gv.nboxes + 1
                if f0[j] <= f0[j+1]:
                    level0 = s+1
                else:
                    level0 = s+2

                # the box with the smaller function value gets level
                # s + 1, the one with the larger function value gets
                # level s + 2
                ipar, level, ichild, f = genbox(
                    ipar, level, ichild, f,
                    par, level0, -nchild, f0[j])
                updtrec(gv.nboxes, level[gv.nboxes], f[0, :])
                # if prt > 1
            else:
                x[i] = x0[i, j]
                gv.nbasket = gv.nbasket + 1
                xmin[:, gv.nbasket] = x
                fmi[nbasket] = f0[j]
                # if prt > 1

            nchild = nchild + 1
            if f0[j+1] < f0[j] < s + 2 < smax:
                nboxes = nboxes + 1
                if f0[j+1] < f0[j]:
                    level0 = s + 1
                else:
                    level0 = s + 2

                ipar, level, ichild, f = genbox(
                    ipar, level, ichild, f,
                    par, level0, -nchild, f0[j+1])
                updtrec(gv.nboxes, level(gv.nboxes), f[0, :])
                # if prt > 1
            else:
                x[i] = x0[i, j + 1]
                gv.nbasket = gv.nbasket + 1
                xmin[:, gv.nbasket] = x
                fmi[gv.nbasket] = f0[j+1]
                # if prt > 1

    else:
        # if prt > 1 ...
        for j in range(L[i]):
            x[i] = x0[i, j]
            nbasket = gv.nbasket + 1
            xmin[:, gv.nbasket] = x
            fmi[gv.nbasket] = f0[j]
            # if prt > 1 ...
            # if prt > 1

    return xbest, fbest, f0, xmin, fmi, ipar, level, ichild, f, flag, ncall

