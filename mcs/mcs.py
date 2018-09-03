import numpy as np
import math
import sys

from .defaults import eps
from .basket import basket
from .exgain import exgain
from . import global_val as gv
from .util import find, add_value, sort_idx
from .subint import subint
from .strtsw import strtsw
from .init import init
from .initbox import initbox
from .vertex import vertex
from .splrnk import splrnk
from .update import updtrec
from .check import chrelerr, chvtr
from .split import split
from .loc import chkloc, addloc
from .lsearch import lsearch

import pdb


def __set_defaults(n, u, v, smax, nf, stop, local,
                   iinit, gamma, hess):
    if not smax:
        smax = 5 * n + 10

    if not nf:
        nf = 50 * n ^ 2

    if not stop:
        stop = np.array([3*n, 0, 0])

    if not iinit:
        if abs(sum(u)) == math.inf:
            iinit = 1
        elif abs(sum(v)) == math.inf:
            iinit = 1
        else:
            iinit = 0

    if not local:
        local = 50

    if not gamma:
        gamma = eps

    if not hess:
        hess = np.ones((n, n))

    return smax, nf, stop, local, iinit, gamma, hess


def __set_init(iinit, n, u, v):
    x0 = np.zeros((u.size, 3))
    if iinit == 0:
        x0[:, 0] = u[:, 0]
        x0[:, 1] = (u[:, 0]+v[:, 0]) / 2
        x0[:, 2] = v[:, 0]

    elif iinit == 1:
        for i in range(1, n):
            if u[i] >= 0:
                x0[i, 0] = u[i]
                x0[i, 1], x0[i, 2] = subint(u[i], v[i])
                x0[i, 1] = 0.5 * (x0[i, 0] + x0[i, 2])
            elif v[i] <= 0:
                x0[i, 0] = v[i]
                x0[i, 1], x0[i, 2] = subint(v[i], u[i])
                x0[i, 1] = 0.5 * (x0[i, 0] + x0[i, 2])
    elif iinit == 2:
        x0[:, 0] = (5*u + v) / 6
        x0[:, 1] = 0.5*(u+v)
        x0[:, 2] = (u + 5*v) / 6
    elif iinit == 3:
        raise NotImplemented(
            "iinit 3 (line searches) not implemented")

    if np.isinf(x0).any():
        raise ValueError("infinates in initialisation list")

    l = 2 * np.ones((n, 1), dtype=int)
    L = 3 * np.ones((n, 1), dtype=int)

    return x0, l, L


def mcs(fun, data, u, v, smax=None, nf=None,
        stop=None, local=None, iinit=None,
        gamma=None, hess=None):

    if not isinstance(u, np.ndarray):
        u = np.array(u)
    if not isinstance(v, np.ndarray):
        v = np.array(v)

    n = u.size

    smax, nf, stop, local, iinit, gamma, hess = __set_defaults(
        n, u, v, smax, nf, stop, local, iinit, gamma, hess)

    if u[v < u]:
        raise ValueError("incompatible box bounds")
    elif u[u == v]:
        raise ValueError("degenerate box bound")

    nloc = 0;
    step1 = 10000; step = 1000; dim = step1

    isplit = np.zeros((1, step1))
    level = np.zeros((1, step1))
    ipar = np.zeros((1, step1))
    ichild = np.zeros((1, step1))
    f = np.zeros((2, step1))
    z = np.zeros((2, step1))
    nogain = np.zeros((step1))

    x0, l, L = __set_init(iinit, n, u, v)

    f0, istar, ncall1 = init(fun, data, x0, l, L, n)
    gv.ncall = gv.ncall + ncall1

    x = np.array(
        [x0[i, l[i]-1][0] for i in range(n)]).transpose()

    v1 = np.zeros(n)

    for i in range(n):
        if abs(x[i]-u[i]) > abs(x[i] - v[i]):
            v1[i] = u[i]
        else:
            v1[i] = v[i]

    gv.record = np.zeros(smax-1)
    gv.record[0] = 1

    gv.xloc = np.array([])
    m = n
    nloc = 0
    flag = 1
    nbasket0 = 0

    xmin = np.zeros((n, 1))
    fmi = np.array([])

    ipar, level, ichild, f, isplit, p, xbest, fbest = initbox(
        x0, f0, l, L, istar, u, v)
    f0min = fbest

    if stop[0] > 0 and stop[0] < 1:
        flag = chrelerr(fbest, stop)
    elif stop[0] == 0:
        flag = chvtr(fbest, stop[1])

    if not flag:
        return xbest, fbest, xmin, fmi, gv.ncall, gv.ncloc, flag

    s = strtsw(smax, level, f)

    gv.nsweep = gv.nsweep + 1

    lc = 0
    while s < smax and gv.ncall + 1 <= nf:
        lc += 1
        par = gv.record[s-1][0]
        print("loopCount: {}".format(lc))
        # pdb.set_trace()
        # compute the base vertex x, the opposite vertex y, the
        # 'neighboring' vertices and their function values needed
        # for quadratic interpolation and the vector n0 indicating
        # that the ith coordinate has been split n0(i) times in
        # the history of the box
        n0, x, y, x1, x2, f1, f2 = vertex(par, n, u, v,
                                          v1, x0, f0, ipar,
                                          isplit, ichild,
                                          z, f, l, L)
        # s 'large'
        if s > 2 * n * (min(n0) + 1):
            # splitting index and splitting value z(2,par) for
            # splitting by rank are computed
            # z(2,par) is set to Inf if we split according to the
            # init. list
            isplit_val, z_val = splrnk(n, n0, p, x, y)
            isplit = add_value(isplit, par, isplit_val)
            z = add_value(z, (1, par), z_val)
            # indicates the box is to be split
            splt = 1
        else:
            # box has already been marked as not eligible for
            # splitting by expected gain
            if nogain[par]:
                splt = 0
            else:
                # splitting by expected gain
                # compute the expected gain vector e and the
                # potential splitting index and splitting value
                f_val = f[par] if len(f.shape) == 1 else f[0, par]
                e, isplit_val, z[1, par] = exgain(
                    n, n0, l, L, x, y, x1, x2, f_val,
                    f0, f1, f2)
                isplit = add_value(isplit, par, isplit_val)
                fexp = f_val + min(e)
                if fexp < fbest:
                    splt = 1
                else:
                    splt = 0
                    nogain[par] = 1

        # prepare for splitting
        if splt == 1:
            i = isplit[par] - 1
            level[par] = 0
            # splitting by init list
            if z[1, par] == math.inf:
                m = m + 1
                z[2, par] = m
                xbest, fbest, f0[:, m], xmin, ipar,
                level, ichild, f, flag,  = splinit(
                    fun, data, i, s, smax, par, x0, n0, u, v,
                    x, y, x1, x2, L, l, xmin, fmi, ipar, level,
                    ichild, f, xbest, fbest, stop)
                gv.ncall = gv.ncall + ncall1
            # default split
            else:
                z[0, par] = x[i]
                [xbest, fbest, xmin, fmi, ipar, level, ichild,
                 f, flag, ncall1] = split(fun, data, i, s, smax, par,
                                          n0, u, v, x, y, x1, x2,
                                          z[:, par], xmin, fmi, ipar,
                                          level, ichild, f, xbest,
                                          fbest, stop)
                gv.ncall = gv.ncall + ncall1

            # if the pre-assigned size of the `large' arrays has
            # already been exceeded, these arrays are made larger
            if gv.nboxes > dim:
                raise NotImplemented()
                isplit[gv.nboxes+1:gv.nboxes+step] = np.zeros(step)
                level[gv.nboxes+1:gv.nboxes+step] = np.zeros(step)
                ipar[gv.nboxes+1:gv.nboxes+step] = np.zeros(step)
                ichild[gv.nboxes+1:gv.nboxes+step] = np.zeros(step)
                z[:, nboxes+1:gv.nboxes+step] = np.zeros((2, step))
                nogain[gv.nboxes+1:nboxes+step] = np.zeros(step)
                f[:, nboxes+1:nboxes+step] = np.zeros((2, step))
                dim = gv.nboxes + step

            if not flag:
                return xbest, fbest, xmin, fmi, gv.ncall, nloc, flag

        else:
            if s + 1 < smax:
                level = add_value(level, par, s + 1)
                # s+1 in MATLAB
                updtrec(par, s, f if len(f.shape) == 1 else f[0, :])
            else:
                level = add_value(level, par, 0)
                gv.nbasket = gv.nbasket + 1
                xmin[:, gv.nbasket] = x
                fmi[gv.nbasket] = f[par] if len(f.shape) == 1 else f[0, par]

            # if prt > 1

        s = s + 1
        while s < smax:
            if gv.record[s-1] == 0:
                s = s + 1
            else:
                break

        # if smax is reached, a new sweep is started
        if s == smax:
            # if doing local search
            if local:
                fmi[nbasket0:gv.nbasket], j = sort_idx(
                        fmi[nbasket0:gv.nbasket]
                    )
                j_val = [q + nbasket0 for q in j]
                xmin[:, nbasket0:gv.nbasket] = xmin[:, j_val]
                xmin0 = []
                fmi0 = []

                for j in range(nbasket0, gv.nbasket):
                    x = xmin[:, j]
                    x = x.reshape((x.size, 1))
                    f1 = fmi[j]
                    loc = chkloc(x, nloc)

                    if loc:
                        nloc = addloc(x, nloc)
                        [xbest, fbest, xmin, fmi, x, f1, loc,
                         flag, ncall1] = basket(
                            fun, data, x, f1, xmin, fmi, xbest,
                            fbest, stop, nbasket0)
                        gv.ncall = gv.ncall + ncall1

                        if not flag:
                            return (xbest, fbest, xmin,
                                    fmi, gv.ncall, nloc, flag)

                        if loc:
                            pdb.set_trace()
                            xmin1, fmi1, nc, flag = lsearch(
                                fun, data, x, f1, f0min, u, v,
                                nf-gv.ncall, stop, local, gamma,
                                hess
                            )

            s = strtsw(smax, level, f[:])

            if stop[0] > 1:
                if gv.nsweep - gv.nsweepbest >= stop[0]:
                    flag = 3
                    return (xbest, fbest, xmin, fmi,
                            gv.ncall, gv.nloc, flag)
            gv.nsweep = gv.nsweep + 1

    if gv.ncall >= nf:
        flag = 2

    if local:
        pass

    return xbest, fbest, xmin, fmi, ncall, ncloc, flag
