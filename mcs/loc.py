import numpy as np
import math

from .util import add_column
from . import global_val as gv

def chkloc(x, nloc):
    for k in range(nloc):
        if x == gv.xloc[:, k]:
            return False

    return True


def addloc(x, nloc):
    gv.xloc = add_column(gv.xloc, nloc, x)
    return nloc + 1
