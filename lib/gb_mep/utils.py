#!/usr/bin/env python3
import numpy as np

## Combination of dictionaries
def combine_dictionaries(d1, d2, cut_d2=0, add_in_between=None):
	combined_dictionary = {}
	for node in np.union1d(list(d1.keys()), list(d2.keys())):
		if hasattr(d1[node],'x') and hasattr(d2[node],'x'):
			if add_in_between is not None:
				combined_dictionary[node] = np.concatenate((d1[node].x, [add_in_between], d2[node].x[cut_d2:]))
			else:
				combined_dictionary[node] = np.concatenate((d1[node].x, d2[node].x[cut_d2:]))
		else:
			if add_in_between is not None:
				combined_dictionary[node] = np.concatenate((d1[node], d2[node][cut_d2:]))
			else:
				combined_dictionary[node] = np.concatenate((d1[node], [add_in_between], d2[node][cut_d2:]))
	return combined_dictionary

## Append to dictionary
def append_to_dictionary(d, val):
	out_dict = {}
	for node in d:
		if hasattr(d[node],'x'):
			out_dict[node] = np.insert(d[node].x, -1, val)
		else:
			out_dict[node] = np.insert(d[node], -1, val)
	return out_dict

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
	## Return model parameters in transformed form
	return params