import numpy as np
import math
import pdb

from .defaults import eps
from .util import find, add_value
from .polint import polint

from .gls.gls import gls


def csearch(fun, data, x, f, u, v, hess=None):
    n = len(x)
    x = np.minimum(v, np.maximum(x, u))

    nohess = False
    if hess is None:
        nohess = True
        hess = np.ones((n, n))

    nfcsearch = 0
    smaxls = 6
    small = 0.1
    nloc = 1
    xmin = x
    fmi = f
    xminnew = xmin
    fminew = fmi
    x1 = np.array([])
    x2 = np.array([])
    f1 = 0
    f2 = 0

    g = np.zeros((n, 1))
    ind0 = np.array([])

    for i in range(n):
        p = np.zeros((n, 1))
        p[i, 0] = 1

        if xmin[i, 0]:
            delta = math.pow(eps, 1/3) * abs(xmin[i])
        else:
            delta = math.pow(eps, 1/3)

        linesearch = 1
        if xmin[i, 0] <= u[i]:
            f1 = fun(data, xmin+delta*p)
            nfcsearch += 1
            if f1 >= fmi:
                f2 = fun(data, xmin+2*delta*p)
                nfcsearch += 1
                x1 = add_value(x1, i, xmin[i, 0] + delta)
                x2 = add_value(x2, i, xmin[i, 0] + 2 * delta)

                if f2 >= fmi:
                    xminnew[i, 0] = xmin[i, 0]
                    fminew = fmi
                else:
                    xminnew[i, 0] = x2[i, 0]
                    fminew = f2

                linesearch = 0
            else:
                alist = np.array([0, delta])
                flist = np.array([fmi, f1])

        elif xmin[i] >= v[i]:
            f1 = fun(data, xmin-2*delta*p)
            nfcsearch += 1
            if f1 >= fmi:
                x1 = add_value(x1, i, xmin[i] - delta)
                x2 = add_value(x2, i, xmin[i] - 2 * delta)
                nfcsearch += 1

                if f2 >= fmi:
                    xminnew[i] = xmin[i]
                    fminew = fmi
                else:
                    xminnew[i] = x2[i]
                    fminew = f2

                linesearch = 0
            else:
                alist = np.array([0, -delta])
                flist = np.array([fmi, f1])
        else:
            alist = 0
            flist = fmi

        if linesearch:
            pdb.set_trace()
            alist, flist, nfls = gls(fun, data, u, v,
                                     xmin, p, alist, flist,
                                     nloc, small, smaxls)
            nfcsearch += nfls
            fminew, j = min(flist)

            if fminew == fmi:
                j = find(alist, lambda x: x == 0)

            ind = find(abs(alist-alist[j]), lambda x: x < delta)
            ind1 = find(ind, lambda x: x == j)

            ind[ind1] = []
            alist[ind] = []
            flist[ind] = []
            fminew, j = min(flist)
            xminnew[i] = xmin[i] + alist[j]

            if i == 0 or not alist[j]:
                if j == 1:
                    x1 = add_value(x1, i, xmin[i] + alist[1])
                    f1 = flist[1]
                    x2 = add_value(x2, i, xmin[i] + alist[2])
                    f2 = flist[2]
                elif j == len(alist):
                    x1 = add_value(x1, i, xmin[i] + alist[j-1])
                    f1 = flist[j-1]
                    x2 = add_value(x2, i, xmin[i] + alist[j-2])
                    f2 = flist[j-2]
                else:
                    x1 = add_value(x1, i, xmin[i] + alist[j-1])
                    f2 = flist[j-1]
                    x2 = add_value(x2, i, xmin[i] + alist[j+1])
                    f2 = flist[j+1]

                xmin = add_value(xmin, i, xminnew[i])
                fmi = fminew
            else:
                x1 = add_value(x1, i, xminnew[i])
                f1 = fminew
                if xmin[i] < x1[i] & j < len(alist):
                    x2 = add_value(x2, i, xmin[i] + alist[j+1])
                    f2 = flist[j+1]
                elif j == 1:
                    if alist[j+1]:
                        x2 = add_value(x2, i, xmin[i] + alist[j+1])
                        f2 = flist[j+1]
                    else:
                        x2 = add_value(x2, i, xmin[i] + alist[j+2])
                        f2 = flist[j+2]
                elif alist[j-1]:
                    x2 = add_value(x2, i, xmin[i] + alist[j-1])
                    f2 = flist[j-1]
                else:
                    x2 = add_value(x2, i, xmin[i] + alist[j-2])
                    f2 = flist[j-2]

        g_val, G_val = polint([xmin[i], x1[i], x2[i]],
                              [fmi, f1, f2])
        g = add_value(g, i, g_val)
        G = add_value(G, (i, i), G_val)
        x = xmin
        k1 = 0

        if f1 <= f2:
            x[i] = x1[i]
        else:
            x[i] = x2[i]

        for k in range(i-1):
            if hess[i, k]:
                q1 = fmi + g[k] * (x1[k]-xmin[k])+0.5*G[k, k]*math.pow((x1[k] - xmin[k]), 2)
                q2 = fmi + g[k] * (x2[k]-xmin[k])+0.5*G[k, k]*math.pow((x2[k] - xmin[k]), 2)

                if q1 <= q2:
                    x[k] = x1[k]
                else:
                    x[k] = x2[k]

                f12 = fun(data, x)
                nfcsearch += 1
                G = add_value(
                        G, (i, k),
                        hessian(i, k, x, xmin, f12, fmi, g, G))
                G = add_value(G, (k, i), G[i, k])

                if f12 < fminew:
                    fminew = f12
                    xminnew = x
                    k1 = k

                x = add_value(x, k, xmin[k])
            else:
                G = add_value(G, (i, k), 0)
                G = add_value(G, (k, i), 0)

        if fminew <= fmi:
            if x1[i] == xminnew[i]:
                x1[i] = xmin[i]
            elif x2[i] == xminnew[i]:
                x2[i] = xmin[i]

            if k1 > 0:
                if xminnew[k1] == x1[k1]:
                    x1[k1] = xmin[k1]
                elif xminnew[k1] == x2[k1]:
                    x2[k1] = xmin[k1]

            for k in range(i):
                g = add_value(
                        g, i,
                        g[k] + G[i, k] * (xminnew[i] - xmin[i])
                    )
                if nohess and k1 > 0:
                    g = add_value(
                        g, k,
                        g[k] + G[k1, k] * (xminnew[k1] - xmin[k1])
                    )


        xmin = xminnew
        fmi = fminew

    return xmin, fmi, g, G, nfcsearch


