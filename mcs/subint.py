import numpy as np
import math

def subint(x, y):
	x2 = y
	f = 1000

	if f*abs(x) < 1:
		if abs(y) > f:
			x2 = np.sign(y)
	else:
		if abs(y) > f * abs(x):
			x2 = 10 * np.sign(y) * abs(x)

	x1 = x + (x2 - x) / 10
	return x1, x2