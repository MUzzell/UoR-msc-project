import numpy as np
import math


def chrelerr(fbest, stop):
    """
      checks whether the required tolerance for a test function with known
      global minimum has already been achieved
      Input:
      fbest     function value to be checked
      stop(0)   relative error with which a global minimum with not too
            small absolute value should be reached
      stop(1)   global minimum function value of a test function
      stop(2)   if abs(fglob) is very small, we stop if the function
            value is less than stop(2)
      Output:
      flag          = 0 the required tolerance has been achieved
            = 1 otherwise
    """

    fglob = stop[1]

    if fbest - fglob <= max(stop[0] * abs(fglob), stop[2]):
        return 0

    return 1


def chvtr(f, vtr):
    '''
      checks whether a required value to reach has already been reached; in
      that case flag is set to 0, otherwise it is set to 1
      Input:
      f function value to be checked
      vtr   value to reach
      Output:
      flag  = 0  vtr has been reached
        = 1  otherwise
    '''
    if f <= vtr:
        return 0

    return 1
