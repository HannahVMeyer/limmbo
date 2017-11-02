######################
### import modules ###
######################

import sys
sys.path.append('./../../')

import scipy as sp
import scipy.linalg as la
import numpy as np
import pandas as pd
from distutils.util import strtobool
#from mtSet.pycore.utils.normalization import gaussianize
from scipy_sugar.stats import quantile_gaussianize


####################################
### functions: data manipulation ###
####################################
def boolanize(string):
    """ 
    Convert command line parameter "True"/"False" into bool
    Input:
        * string: "False" or "True" [string]
    Output:
        * False/True [bool]
    """
    return bool(strtobool(string))


def nans(shape):
    """ 
    Create np.array of nans
    Input:
        * shape: [int] or [tuple] of [int] with shape of the empty array
    Output:
        * np.array of NaNs
    """
    a = np.empty(shape, dtype=float)
    a.fill(np.nan)
    return a


def scale(x):
    """ 
    Mean center and unit variance input array
    Input:
        * x: np.array
    Output:
        mean-centered, unit-variance np.array of x
    """
    x -= x.mean(axis=0)
    x /= x.std(axis=0)
    return x


def transform(x, type="scale"):
    """ 
    Transform input array:
        * scale: mean-center, unit variance
        * gaussian: inverse normalise
        * None: No transformation
    Input: 
        * x: np.array of data to transform
        * scale: name [string] of transformation method (scale,gaussian,None)
    Output:
        * transformed np.array of x
    """
    if type is "scale":
        x = scale(x)
    if type is "gaussian":
        x = quantile_gaussianize(x)
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
        * eigenvalue: np.array of eigenvalues
    Output:
        * v: variance explained [float]
    """
    v = eigenvalue.cumsum()
    v /= v.max()
    return(v)


def regularize(covMatrix, verbose=True):
    """
    Make matrix positive-semi definite by ensuring minimum eigenvalue >= 0:
    add absolute value of minimum eigenvalue and 1e-4 (for numerical stability 
    of abs(min(eigenvalue) < 1e-4 to diagonal of matrix
    Input: 
        * covMatrix: square matrix [np.array]
    Output:
        * covMatrix: positive, semi-definite matrix from input covMatrix
          [np.array]
        * S.min: minimum eigenvalue of input covMatrix
    """
    S, U = la.eigh(covMatrix)
    minS = S.min()
    if minS < 0:
        verboseprint("Regularizing: minimum Eigenvalue %6.4f" % S.min(),
                     verbose=verbose)
        covMatrix += (abs(S.min()) + 1e-4) * sp.eye(covMatrix.shape[0])
    elif minS < 1e-4:
        verboseprint("Make numerically stable: minimum Eigenvalue %6.4f" % \
                S.min(), verbose=verbose)
        covMatrix +=  1e-4 * sp.eye(covMatrix.shape[0])
    else:
        print "No regularizing: minimum Eigenvalue %6.4f" % S.min()
    return(covMatrix, minS)


def generate_permutation(seed=12321, n=1000, P=100, p=10,
                         exclude_zero=False):
    """
    Generate permutation.
    Input:
        * seed: numeric; used as seed for pseudo-random numbers generation; 
          default: 12321
        * n: number [int] of permutations to generated, default: 1000
        * P: total number [int] of traits, default: 100
        * p: subsampling size [int], default: 10
        * exclude_zero: [bool] should zero be in set to draw from
    Output;
        * return_list: list of length n containing [np.arrays] of length [p] 
          with subsets/permutations (if P=p) of numbers range(P)
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
        * bootstrap_traits: [S x S] covariance matrix estimates [np.array]
        * bootstrap: [S x 1] np.array; indices to project [S x S] matrix 
          values into large matrix
        * P: total number [int] of dimensions default: 100
        * zeros: [bool] fill void spaces in large matrix with zeros (True, 
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

def verboseprint(message, verbose=True):
    """
    Print message if verbose option is chosen
    Input:
        * message: text [string} to print
        * verbose: [bool] flag whether to print message (True) or not (False)
    """
    if verbose is True:
        print message

def match(samples_ref, data_compare, samples_compare, squarematrix=False):
    """
    Match the order of data and ID matrices to a reference sample order
    Input:
        * samples_ref: [M] sammple Ids [np.array] used as reference
        * data_compare: [N x L] data matrix with [N] samples and [L] columns 
          [np.array]
        * samples_compare: [N] sample IDs [np.array] to be matched to
         samples_ref
        * squarematrix: [bool] is data_compare a square matrix i.e. samples in 
          cols and rows
    Output:
        * data_compare: [M x L] data matrix of input data_compare [np.array]
        * samples_compare: [M] sample IDs of input samples_compare [np.array]
        * samples_before: numer [int] of samples in data_compare/samples_compare
          before matching to samples_ref
        * samples_after: numer [int] of samples in data_compare/samples_compare
          after matching to samples_ref
    """
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
