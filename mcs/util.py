import numpy as np

from . import global_val as gv


def genbox(ipar, level, ichild, f,
           par, level0, nchild, f0):

    ipar = add_value(ipar, gv.nboxes -1, par)
    level = add_value(level, gv.nboxes - 1, level0)
    ichild = add_value(ichild, gv.nboxes - 1, nchild)
    if len(f.shape) <= 1:
        f = add_value(f, gv.nboxes - 1, f0)
    else:
        f = add_value(f, (0, gv.nboxes - 1), f0)

    return ipar, level, ichild, f


def find(x, func):
    return [i for (i, val) in enumerate(x) if func(val)]


def sort_idx(array, axis=-1):

    def select_key(x):
        if axis == -1:
            return x[1]
        else:
            return x[1][axis]

    array = sorted(enumerate(array), key=select_key)
    return [x[1] for x in array], [int(x[0]) for x in array]


def logi_array(array):

    return np.array([1 if val else 0 for _, val in enumerate(array)])


def expand_np_tuple(array, size):

    #import pdb; pdb.set_trace()

    #if len(size) != 2:
    #    raise ValueError("wot")

    #if len(size) > 2:
    #    raise ValueError("Uh oh, size shape > 2")

    if len(array.shape) == 1:
        shape = (1, array.shape[0])
    else:
        shape = array.shape

    if size[0] + 1 > shape[0]:
        array1 = array.flatten()
        array1 = np.append(array1, [0 for i in range(shape[1] * size[0])])
        array = array1.reshape((size[0] + 1, shape[1]))

    shape = array.shape

    if size[1] + 1 > shape[1]:
        array1 = array.flatten()
        for i in range(size[1] + 1 - shape[1]):
            array1 = np.insert(array1, [shape[1] * i for i in range(1, shape[0]+1)], 0)
        array = array1.reshape((shape[0], size[1] + 1))

    return array

    if len(array.shape) == 1:
        array = np.insert
        array = np.vstack(
            [array] + [np.zeros(array.shape[0]) for i in range(1, size[0]+1)])

    return array


def expand_np(array, newsize):

    if array.size >= newsize:
        return array

    return np.append(array, [0 for i in range(array.size, newsize)])


def add_value(array, idx, value):

    if type(idx) == list:
        idx = tuple(idx)

    if type(idx) == tuple:
        array = expand_np_tuple(array, idx)
    else:
        array = expand_np(array, idx+1)

    array[idx] = value
    return array


def add_column(array, idx, value):

    if type(idx) != int:
        raise ValueError("Can only handle Ints")

    if len(value.shape) == 1 or value.shape[1] == 1:
        value = value.reshape((1, value.size))

    if array.shape == (0,):
        array = np.zeros(value.transpose().shape)


    if len(array.shape) != 2:
        raise ValueError("Can only handle 2D arrays")

    if array.shape[1] <= idx:
        array = np.hstack(
            (array, np.zeros((array.shape[0], idx+1 - array.shape[1]))))

    array[:, idx] = value

    return array


def min_idx(array):

    val, idx = sort_idx(array)

    return val[0], idx[0]


def max_idx(array):

    maxval = max(array)

    n = 0
    for i in array:
        if i == maxval:
            return maxval, n
        n = n+1
