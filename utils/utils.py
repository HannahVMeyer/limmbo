
# coding: utf-8
######################
### import modules ###
######################


import sys
sys.path.append('/homes/hannah/bin/python_modules')
sys.path.append('/homes/hannah/LiMMBo')
sys.path.append(
    '/homes/hannah/software/python2.7.8/lib/python2.7/site-packages')


import scipy as sp
import scipy.linalg as la
import numpy as np
import pandas as pd
from distutils.util import strtobool
from mtSet.pycore.utils.normalization import gaussianize


####################################
### functions: data manipulation ###
####################################
def boolanize(string):
    """ Convert command line parameter "True"/"False" into bool"""
    return bool(strtobool(string))


def nans(shape, dtype=float):
    """ Create [shape] np.array of nans"""
    a = np.empty(shape, dtype)
    a.fill(np.nan)
    return a


def scale(x):
    """ Mean center and unit variance input array"""
    x -= x.mean(axis=0)
    x /= x.std(axis=0)
    return x


def transform(x, type="scale"):
    """ Transform input array: scale/gaussianize/None """
    if type is "scale":
        x = scale(x)
    if type is "gaussian":
        x = gaussianize(x)
    return(x)


def getEigen(covMatrix, reverse=True):
    """ Get eigenvectors and values of hermitian covMatrix:
        * reverse: if True (default): order eigenvalues (and vectors) in 
        decreasing order
    """
    S, U = la.eigh(covMatrix)
    if reverse == True:
        S = S[::-1]
        U = U[:, ::-1]
    return(S, U)


def getVariance(eigenvalue):
    """
    Based on input eigenvalue computes cumulative sum and normalizes to overall 
    sum to obtain variance explained
    Input:
        * np.array of eigenvalues
    """
    v = eigenvalue.cumsum()
    v /= v.max()
    return(v)


def regularize(covMatrix, verbose=True):
    """ Regularize covMatrix if minimum eigenvalue less than 1e-4:
        * add absolute value of minimum eigenvalue and 1e-4 to diagonal of 
        matrix
    """
    S, U = la.eigh(covMatrix)
    if S.min() < 1e-4:
        verboseprint("Regularizing: minimum Eigenvalue %6.4f" % S.min(),
                     verbose=verbose)
        covMatrix += (abs(S.min()) + 1e-4) * sp.eye(covMatrix.shape[0])
    else:
        print "No regularizing: minimum Eigenvalue %6.4f" % S.min()
    return(covMatrix, S.min())


def generate_permutation(seed=12321, n=1000, P=100, p=10,
                         exclude_zero=False):
    """
    Generate permutation.
    Input:
        * seed: numeric; used as seed for pseudo-random numbers generation; 
          default: 12321
        * n: numeric; many permutation should be generated, default: 1000
        * P: numeric; total number of traits, default: 100
        * p: numeric; how small should the permutation subset be, default: 10
    Output;
        * list of length n containing np.array of length p with permutation of 
          numbers range(P)
    """
    rand_state = np.random.RandomState(seed)
    return_list = [None] * n
    if exclude_zero:
        rangeP = range(P)[1:]
    else:
        rangeP = range(P)
    for i in xrange(n):
        perm_dic = rand_state.choice(a=rangeP, size=p, replace=False)
        return_list[i] = perm_dic
    return return_list


def inflate_matrix(bootstrap_traits, bootstrap, P=100, zeros=True):
    """ 
    Project small matrix into large matrix using indeces provided:
    Input:
        * bootstrap_traits: [p x p] np.array; small matrix to be projected
        * bootstrap: list/[p x 1] np.array; indices to project small matrix 
          values into large matrix
        * P: numeric; dimensions of large square matrix; default: 100
        * zeros: bool; fill void spaces in large matrix with zeros (True, 
          default) or nans (False)
    Output:
        * all_traits: large matrix containing small matrix values at indeces 
          and zeros/nans elswhere
    """
    index = np.ix_(np.array(bootstrap), np.array(bootstrap))
    if zeros is True:
        all_traits = np.zeros((P, P))
    else:
        all_traits = nans((P, P))
    all_traits[index] = bootstrap_traits
    return(all_traits)


def ttcooccurence((nrtraits, nrtraitssampled)):
    ttc = 1. / nrtraits * 1. / (nrtraits - 1) * \
        nrtraitssampled * (nrtraitssampled - 1)
    return ttc


def verboseprint(message, verbose=True):
    if verbose is True:
        print message


def match(samples_ref, samples_compare, data_compare, squarematrix=False):
    samples_before = samples_compare.shape[0]
    subset = pd.match(samples_ref, samples_compare)

    data_compare = data_compare[subset, :]
    if squarematrix:
        data_compare = data_compare[:, subset]
    samples_compare = samples_compare[subset]
    samples_after = samples_compare.shape[0]
    np.testing.assert_array_equal(samples_ref, samples_compare,
                                  err_msg=("Col order does not match. These"
                                  "are the differing columns:\t%s")
                                  % (np.array_str(np.setdiff1d(samples_ref, 
                                      samples_compare))))
    return (data_compare, samples_compare, samples_before, samples_after)
