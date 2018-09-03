import numpy as np
import math

from .check import chrelerr, chvtr
from .util import genbox, add_value, add_column
from . import global_val as gv
from .update import updtrec, updtoptl


def split1(x1, x2, f1, f2):
    if f1 <= f2:
        return x1 + 0.5 * (-1 + math.sqrt(5)) * (x2 - x1)
    else:
        return x1 + 0.5 * (3 - math.sqrt(5)) * (x2 - x1)


def split2(x, y):
    """
    Determines a value x1 for splitting the interval [min(x,y),max(x,y)]
    is modeled on the function subint with safeguards for infinite y
    """
    x2 = y
    if x == 0 and abs(y) > 1000:
        x2 = np.sign(y)
    elif x != 0 and abs(y) > 100 * abs(x):
        x2 = np.multiply(10, y) * abs(x)

    return x + 2 * (x2 - x) / 3


def split(fun, data, i, s, smax, par, n0, u, v, x, y, x1, x2, z, xmin,
          fmi, ipar, level, ichild, f, xbest, fbest, stop):
    """
    splits box # par at level s in its ith coordinate into two or three
    children and inserts its children and their parameters in the list
    """
    ncall = 0
    n = len(x)
    iopt = []

    # if prt > 1 goes here

    flag = 1
    x[i] = z[1]

    f = add_value(f, (1, par), fun(data, x))
    ncall = ncall + 1

    if f[1, par] < fbest:
        fbest = f[1, par]
        xbest = x.copy()
        gv.nsweepbest = gv.nsweep

        if stop[0] > 0 and stop[0] < 1:
            flag = chrelerr(fbest, stop)
        elif stop[0] == 0:
            flag = chvtr(fbest, stop)

        if not flag:
            return xbest, fbest, xmin, fmi, ipar, level, ichild, f, flag, ncall

    splval = split1(z[0], z[1], f[0, par], f[1, par])
    if s + 1 < smax:
        if f[0, par] <= f[1, par]:
            gv.nboxes = gv.nboxes + 1
            ipar, level, ichild, f = genbox(
                ipar, level, ichild, f,
                par+1, s+1, 1, f[0, par])
            updtrec(gv.nboxes-1, level[gv.nboxes-1]-1, f[0, :])

            # if prt > 1

            if s + 2 < smax:
                gv.nboxes = gv.nboxes + 1
                ipar, level, ichild, f = genbox(
                    ipar, level, ichild, f,
                    par+1, s + 2, 2, f[1, par])
                updtrec(gv.nboxes-1, level[gv.nboxes-1]-1, f[0,:])
            else:
                x[i] = z[1]
                gv.nbasket = gv.nbasket + 1
                xmin = add_column(xmin, gv.nbasket-1, x)
                fmi = add_value(fmi, gv.nbasket-1, f[1, par])

            # if prt > 1
        else:
            if s + 2 < smax:
                gv.nboxes = gv.nboxes + 1
                ipar, level, ichild, f = genbox(
                    ipar, level, ichild, f,
                    par+1, s + 2, 1, f[0, par])
                updtrec(gv.nboxes-1, level[gv.nboxes-1]-1, f[0, :])
            else:
                x[i] = z[0]
                gv.nbasket = gv.nbasket + 1
                xmin = add_column(xmin, gv.nbasket-1, x)
                fmi = add_value(fmi, gv.nbasket-1, f[0, par])

            # if prt > 1

            gv.nboxes = gv.nboxes + 1
            ipar, level, ichild, f = genbox(
                ipar, level, ichild, f,
                par+1, s + 1, 2, f[1, par])
            updtrec(gv.nboxes-1, level[gv.nboxes-1]-1, f[0, :])

            # if prt > 1

        # if the third box is larger than the smaller of the
        # other two boxes, it gets level s + 1;
        # otherwise it gets level s + 2
        if z[1] != y[i]:
            if abs(z[1] - y[i]) > abs(z[1] - z[0]) * (3 - math.sqrt(5)) * 0.5:
                gv.nboxes = gv.nboxes + 1
                ipar, level, ichild, f = genbox(
                    ipar, level, ichild, f,
                    par+1, s + 1, 3, f[1, par])
                updtrec(gv.nboxes-1, level[gv.nboxes-1]-1, f[0, :])
                # if prt > 1
            else:
                if s + 2 < smax:
                    gv.nboxes = gv.nboxes + 1
                    ipar, level, ichild, f = genbox(
                        ipar, level, ichild, f,
                        par+1, s + 2, 3, f[1, par])
                    updtrec(gv.nboxes-1, level[gv.nboxes-1]-1, f[0, :])
                else:
                    x[i] = z[1]
                    gv.nbasket = gv.nbasket + 1
                    xmin = add_column(xmin, gv.nbasket-1, x)
                    fmi = add_value(fmi, gv.nbasket-1, f[1, par])

                # if prt > 1

    else:
        x[i] = z[0]
        gv.nbasket = gv.nbasket + 1
        xmin = add_column(xmin, gv.nbasket-1, x)
        fmi = add_value(fmi, gv.nbasket-1, f[0, par])
        x[i] = z[1]
        gv.nbasket = gv.nbasket + 1
        xmin = add_column(xmin, gv.nbasket-1, x)
        fmi = add_value(fmi, gv.nbasket-1, f[1, par])

        # if prt > 1

    return xbest, fbest, xmin, fmi, ipar, level, ichild, f, flag, ncall
