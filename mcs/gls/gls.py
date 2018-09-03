import numpy as np
import math
import pdb

from mcs.util import find, sort_idx, logi_array, min_idx

def __lssort(alist, flist):
    pdb.set_trace()

    alist, perm = sort_idx(alist)
    flist = flist[perm]

    if isinstance(alist, list) or len(alist.shape) == 1:
        s = len(alist)
    else:
        # Maybe this should be 0?
        s = alist.shape[1]

    up = logi_array(flist[0: s-1] < flist[1 : s])
    down = logi_array(flist[1:s] <= flist[0:s-1])

    a = flist[s-1] < flist[s-2]
    down[s-2] = 1 if a else 0
    monotone = logi_array([sum(up) == 0 or sum(down) == 0])

    minima = logi_array(
        np.logical_and(np.append(up, [1]), np.append([1], down))
    )
    nmin = sum(minima)

    fbest, i = min_idx(flist)
    abest = alist[i]

    fmed = np.median(flist)

    if nmin > 1:
        al = alist[minima]
        al[al == abest] = []
        unitlen = min(abs(al-abest))
    else:
        unitlen = max(abest-alist[0], alist[s-1] - abest)

    return alist, flist, s, monotone, unitlen




def __lsinit(alist, flist, amin, amax, fun, data, x, p, scale):

    if not isinstance(alist, int) and alist.shape[1] == 0:
        alp = 0
        if amin > 0:
            alp = amin
        if amax < 0:
            alp = amax

        falp = fun(data, x+alp*p)
        alist = alp
        flist = falp
    elif isinstance(alist, int) or alist.shape[1] == 1:
        alp = 0
        if amin > 0:
            alp = amin
        if amax < 0:
            alp = amax
        if alist != alp:
            falp = fun(data, x + alp*p)
            alist = np.array([alist, alp])
            flist = np.array([flist, falp])

    aamin = alist if isinstance(alist, int) else min(alist)
    aamax = alist if isinstance(alist, int) else max(alist)

    if amin > aamin or amax < aamax:
        raise ValueError("Non-admissible step in alist")

    if aamax-aamin <= scale:
        alp1 = max([amin, min([-scale, amax])])
        alp2 = max([amin, min([+scale, amax])])
        alp = math.inf

        if aamin - alp1 >= alp2-aamax:
            alp = alp1
        if alp2 - aamax >= aamin-alp1:
            alp = alp2

        if alp < aamin or alp > aamax:
            falp = fun(data, x+alp * p)
            alist = np.array([alist, alp])
            flist = np.array([flist, falp])

    if (len(alist.shape) == 1 and alist.size <= 1) or alist.shape[0] == 1:
        raise ValueError("No Second point found")

    return __lssort(alist, flist)


def __lsrange(bend, p, x, xl, xu):

    if max(abs(p)) == 0:
        raise ValueError('zero search direction in line search')

    pp = abs(p[p != 0])
    u = np.divide(abs(x[p != 0]), pp)
    scale = min(u)

    if scale == 0:
        u[u == 0] = np.divide(1, pp[u==0])
        scale = min(u)

    if not bend:
        amin = -math.inf
        amax = +math.inf

        for i in range(x.shape[0]):
            if p[i] > 0:
                amin = max(amin, (xl[i]-x[i]) / p[i])
                amax = min(amax, (xu[i]-x[i]) / p[i])
            elif p[i] < 0:
                amin = max(amin, (xu[i]-x[i]) / p[i])
                amax = min(amin, (xl[i]-x[i]) / p[i])

        if amin>amax:
            raise ValueError("No adminssible step in line search")

        # if printing

    else:
        amin = +math.inf
        amax = -math.inf

        for i in range(x, 1):
            if p[i] > 0:
                amin = max(amin, (xl[i]-x[i, 0]) / p[i])
                amax = min(amax, (xu[i]-x[i, 0]) / p[i])
            elif p[i] < 0:
                amin = max(amin, (xu[i]-x[i]) / p[i])
                amax = min(amin, (xl[i]-x[i]) / p[i])

    return amin, amax, pp, u, scale


def gls(fun, data, xl, xu, x, p, alist, flist,
        nloc=1, small=0.1, smax=10):

    short = 0.381966

    if isinstance(alist, int):
        sinit = 1
    else:
        sinit = alist.shape[1]

    bend = 0;

    amin, amax, pp, u, scale = __lsrange(bend, p, x, xl, xu)

    alist, flist, s, monotone, unitlen = __lsinit(
        alist, flist, amin, amax, fun, data, x, p, scale
    )

    nf = s - sinit

    while s < min(5, smax):
        if nloc == 1:
            

