#!/usr/bin/env python3
import numpy as np

## Parameter transformations
def transform_parameters(p, to_unconstrained=False):
	# Dimension of the parameter vector (one if only one number is provided)
	if isinstance(p, float) or isinstance(p, int):
		d = 1
	else:
		d = len(p)
	# If the parameters are transformed to unconstrained form, use logarithms
	if to_unconstrained:
		if d == 1:
			params = np.log(p)
		elif d == 3:
			params = np.log(p)
			params[2] = np.log(p[2]-p[1])
		elif d == 4:
			params = np.log(p)
			params[2] = np.log(p[2]-p[1])
		elif d == 5:
			params = np.log(p)
			params[2] = np.log(p[2]-p[1])
			params[4] = np.log(p[4]-p[3])
		elif d == 6:
			params = np.log(p)
			params[2] = np.log(p[2]-p[1])
			params[4] = np.log(p[4]-p[3])
		elif d == 7:
			params = np.log(p)
			params[2] = np.log(p[2]-p[1])
			params[4] = np.log(p[4]-p[3])
		else:
			return ValueError('Incorrect number of parameters.')
	else: # If the parameters are transformed to constrained form, use exponentials
		if d == 1:
			params = np.exp(p)
		elif d == 3:
			params = np.exp(p)
			params[2] += params[1]
		elif d == 4:
			params = np.exp(p)
			params[2] += params[1]
		elif d == 5:
			params = np.exp(p)
			params[2] += params[1]
			params[4] += params[3]
		elif d == 6:
			params = np.exp(p)
			params[2] += params[1]
			params[5] += params[4]
		elif d == 7:
			params = np.exp(p)
			params[2] += params[1]
			params[5] += params[4]
		else:
			return ValueError('Incorrect number of parameters.')