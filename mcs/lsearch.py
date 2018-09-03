import numpy as np
import math

from . import global_val as gv
from .csearch import csearch
from .check import chrelerr, chvtr
from .defaults import eps
from .neighbour import neighbour
from .util import sort_idx, find, add_value
import pdb


def __fpred(fmi, g, p, G):

    a1 = np.multiply(g.transpose(), p)
    a2 = np.multiply(np.multiply(p.transpose(), G), p)

    return np.add(np.add(fmi, a1), np.multiply(0.5, a2))

def __gen_r(fold, fmi, fpred):

    r1 = np.subtract(fold, fmi)
    r2 = np.subtract(fold, fpred)

    return np.divide(r1, r2)


def __keep_running(ncall, nf, nstep, maxstep, diag, ind, n, stop,
                   fmi, gain, b, gamma, f0, f):
    if ncall >= nf or nstep >= maxstep:
        return False

    return (diag or len(ind) < n or
            (stop[0] == 0 and fmi - gain <= stop[1]) or
            (b >= gamma * (f0-f) and gain > 0))

def __min_max(u, v):

    a1 = [max(a, 0) for a in u]

    for i in range(len(v)):
        a1[i] = min(v[i], a1[i])

    return a1


def lsearch(fun, data, x, f, f0, u, v, nf, stop,
            maxstep=50, gamma=eps, hess=None):

    ncall = 0
    n = len(x)
    x0 = np.minimum(np.maximum(u, 0), v)
    flag = 1
    eps0 = 0.001
    nloc = 1
    small = 0.1
    smaxls = 15

    if hess is None:
        hess = np.ones((n, n))

    xmin, fmi, g, G, nfcsearch = csearch(fun, data, x, f,
                                         u, v, hess)
    xmin = max(u, min(xmin, v))
    ncall = ncall + nfcsearch
    xold = xmin
    fold = fmi

    if stop[0] > 0 and stop[0] < 1:
        flag = chrelerr(fmi, stop)
    elif stop[0] == 0:
        flag = chvtr(fmi, stop[1])

    if not flag:
        return xmin, fmi, ncall, flag

    d = min(min(xmin-u, v-xmin), 0.25*(1+abs(x-x0)))
    p = minq(fmi, g, G, -d, d, 0)
    x = max(u, min(xmin+p, v))
    p = x-xmin

    if np.norm(p):
        f1 = fun(data, x)
        ncall = ncall + 1
        alist = [0, 1]
        flist = [fmi, f1]
        fpred = __fpred(fmi, g, p, G)

        alist, flist, nfls = gls(fun, data, u, v, xmin, p,
                                 alist, flist, nloc, small,
                                 smaxls)
        ncall = ncall + nfls
        fminew, i = sort_idx(flist)[-1]

        if fminew == fmi:
            i = find(alist, lambda x: x == 0)
        else:
            fmi = fminew

        xmin = np.add(xmin, np.multiply(alist[i], p))
        xmin = max(u, min(xmin, v))
        gain = f - fmi

        if stop[0] > 0 and stop[0] < 1:
            flag = chrelerr(fmi, stop)
        elif stop[0] == 0:
            flag = chvtr(fmi, stop[1])

        if not flag:
            return xmin, fmi, ncall, flag

        if fold == fmi:
            r = 0
        elif fold == fpred:
            r = 0.5
        else:
            r = __gen_r(fold, fmi, fpred)
    else:
        gain = f - fmi
        r = 0

    diag = 0
    ind = find(xmin, lambda x: u < xmin and xmin < v)
    b = np.multiply(abs(g).transpose(), max(abs(xmin), abs(xold)))
    nstep = 0

    # What the hell?
    while __keep_running(ncall, nf, nstep, maxstep, diag, ind,
                         n, stop, fmi, gain, b, gamma, f0, f):
        nstep += 1
        delta = abs(xmin)*eps ^ (1/3)
        j = find(delta, lambda x: x == 0)

        if not j:
            delta = add_value(
                delta, j,
                np.multiply(math.pow(eps, 1/3),
                            np.ones(j.size))
            )

        x1, x2 = neighbour(xmin, delta, u, v)
        f = fmi

        if (len(ind) < n and
                (b < np.multiply(gamma, f0-f) or
                 not gain)):
            ind = find(xmin, lambda x: u < xmin and xmin < v)