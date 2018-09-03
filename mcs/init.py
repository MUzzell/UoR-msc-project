import numpy as np
import math


def init(fun, data, x0, l, L, n):
    import pdb; pdb.set_trace()

    ncall = 0
    f0 = np.zeros((L[0, 0], n))
    istar = np.zeros(n, dtype=int)

    x = np.array(
        [x0[i, l[i] - 1] for i in range(0, n)])

    f1 = fun(data, x)
    f0[l[0, 0] - 1, 0] = f1
    ncall = ncall + 1

    for i in range(0, n):
        istar[i] = l[i, 0]
        for j in range(0, L[i, 0]):
            if j == l[i] - 1:
                if i != 0:
                    f0[j, i] = f0[istar[i-1] - 1, i-1]
            else:
                x[i] = x0[i, j]

                f0[j, i] = fun(data, x)
                ncall = ncall + 1
                if f0[j, i] < f1:
                    f1 = f0[j, i]
                    istar[i] = j + 1

        x[i] = x0[i, istar[i] - 1]

    return f0, istar, ncall
