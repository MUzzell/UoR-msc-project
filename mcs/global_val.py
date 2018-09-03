import sys


class Globals(object):

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Globals, cls).__new__(
                cls, *args, **kwargs)

            cls._instance.nboxes = 1
            cls._instance.nbasket = 0
            cls._instance.nglob = 0
            cls._instance.nsweep = 0
            cls._instance.ncall = 0
            cls._instance.nsweepbest = 0
            cls._instance.optlevel = 0

            cls._instance.record = None
            cls._instance.xglob = None
            cls._instance.xloc = None
            cls._instance.foptbox = None

        return cls._instance


global g

if 'g' not in globals():
    g = Globals()

sys.modules[__name__] = g