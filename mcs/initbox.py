import numpy as np
import math

import pdb

from . import global_val as gv
from .polint import polint
from .quadops import quadmin, quadpol
from .update import updtoptl
from .split import split1
from .util import expand_np, max_idx, genbox


def initbox(x0, f0, l, L, istar, u, v):
    n = u.size
    ipar = np.array([0], dtype=np.int32)
    level = np.array([1], dtype=np.int32)
    ichild = np.array([1], dtype=np.int32)

    f = np.array(f0[l[0, 0] - 1, 0])

    var = np.zeros((n))
    xbest = np.zeros((n))
    isplit = np.zeros([0], dtype=np.int32)
    p = np.zeros((n), dtype=np.int32)

    par = 1; fbest = 0

    iopt = np.zeros((0, gv.nglob))

    for i in range(0, n):
        isplit = expand_np(isplit, par)
        isplit[par - 1] = -(i+1)
        nchild = 0

        if x0[i, 0] > u[i]:
            gv.nboxes = gv.nboxes + 1
            nchild = nchild + 1

            ipar, level, ichild, f = genbox(
                ipar, level, ichild, f,
                par, level(par)+1, -nchild, f0(0, i))

            updtoptl(i, u[i], x0[i, 0], iopt, level[par]+1, f0[0, i])

        if L[i, 0] == 3:
            v1 = v[i]
        else:
            v1 = x0[i, 2]

        d = polint(x0[i, 0:3], f0[0:3, i])
        xl = quadmin(u[i], v1, d, x0[i,0:3])
        fl = quadpol(xl, d, x0[i, 0:3])
        xu = quadmin(u[i], v1, -d, x0[i, 0:3])
        fu = quadpol(xu, d, x0[i, 0:3])

        if istar[i] == 1:
            if xl < x0[i, 0]:
                par1 = gv.nboxes
                j1 = 0
            else:
                par1 = gv.nboxes + 1
                j1 = 2

        for j in range(0, L[i, 0] - 1):
            gv.nboxes = gv.nboxes + 1
            nchild = nchild + 1

            if f0[j, i] <= f0[j+1, i]:
                s = 1
            else:
                s = 2

            ipar, level, ichild, f = genbox(
                ipar, level, ichild, f,
                par, level[par - 1]+s, -nchild, f0[j, i])

            splval = split1(x0[i, j], x0[i, j+1], f0[j, i], f0[j+1, i])

            updtoptl(i, x0[i, j], splval, iopt, level[par]+1, f0[j, i])

            if j >= 1:
                if istar[i] - 1 == j:
                    if xl <= x0[i, j]:
                        par1 = gv.nboxes - 1
                        j1 = j - 1 + 1
                    else:
                        par1 = gv.nboxes
                        j1 = j + 1 + 1

                if j <= L[i, 0] - 3:
                    d = polint(x0[i, j:j+2], f0[j:j+2, i])

                    if j < L[i, 0]-2:
                        u1 = x0[i, j+2]
                    else:
                        u1 = v[i]

                    xl = quadmin(x0[i, j], u1, d, x0[i, j:j+2])
                    fl = min(quadpol(xl, d, x0[i, j:j+2]), fl)
                    xu = quadmin(x0[i, j], u1, -d, x0[i, j:j+2])
                    fu = max(quadpol(xu, d, x0[i, j:j+2]), fu)

            gv.nboxes = gv.nboxes + 1
            nchild = nchild + 1

            ipar, level, ichild, f = genbox(
                ipar, level, ichild, f,
                par, level[par - 1]+3-s, -nchild, f0[j+1, i])

            updtoptl(i, splval, x0[i, j+1], iopt, level[par]+1, f0[j+1, i])

        if x0[i, L[i, 0] - 1] < v[i]:
            gv.nboxes = gv.nboxes + 1
            nchild = nchild + 1

            ipar, level, ichild, f = genbox(
                ipar, level, ichild, f,
                par, level[par]+1, -nchild, f0[L[i, 0], i])

            updtoptl(i, x0[i, L[i, 0] - 1], v[i], iopt,
                     level[par]+1, f0[L[i, 0] - 1, i])

        if istar[i] == L[i, 0]:
            if x0[i, L[i, 0] - 1] < v[0]:
                if xl <= x0[i, L[i, 0] - 1]:
                    par1 = gv.nboxes - 1
                    j1 = L[i, 0] - 1
                else:
                    par1 = nboxes
                    j1 = L[i, 0] + 1
            else:
                par1 = nboxes
                j1 = L[i, 0] - 1

        var[i] = fu - fl

        level[par - 1] = 0 # box is marked as split
        par = par1
        if j1 == 0:
            splval = u[i]
        elif j1 == L[i, 0] + 1:
            splval = v[i]
        else:
            if j1 < istar[i]:
                splval = split1(x0[i, j1-1], x0[i, istar[i]-1],
                                f0[j1-1, i], f0[istar[i]-1, i])
            else:
                splval = split1(x0[i, istar[i]-1], x0[i, j1-1],
                                f0[istar[i]-1, i], f0[j1-1, i])

        '''
        if i <= n-1:
            iopt1 = []
            for j in range(0, len(iopt)):
                if min(splval, x0[i, istar[i]]) <= gv.xglob[i,iopt[j]]:
                    if gv.xglob[i,iopt[j]] <= max(splval, x0[i, istar[i]]):
                        iopt1 = [iopt1, iopt[j]]
            iopt = iopt1
        '''
    fbest = f0[istar[n-1] - 1, n-1]

    for i in range(0, n):
        var0, p[i] = max_idx(var)
        var[p[i]] = -1
        xbest[i] = x0[i, istar[i] - 1]

    return ipar, level, ichild, f, isplit, p, xbest, fbest
