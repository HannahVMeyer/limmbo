import scipy as sp
import scipy.linalg as la
import scipy.stats as stats
import numpy as np
import pandas as pd
from distutils.util import strtobool
from math import sqrt


def boolanize(string):
    r"""
    Convert command line parameter "True"/"False" into boolean

    Arguments:
        string (string):
        "False" or "True"

    Returns:
        (bool):

            False/True
    """
    return bool(strtobool(string))


def nans(shape):
    r"""
    Create numpy array of NaNs

    Arguments:
        shape (tuple):
            shape of the empty array

    Returns:
        (numpy array):
            numpy array of NaNs
    """
    a = np.empty(shape, dtype=float)
    a.fill(np.nan)
    return a


def scale(x):
    r"""
    Mean center and unit variance input array

    Arguments:
        x (array-like):
            array to be scaled by column
    Returns:
        (numpy array):
            mean-centered, unit-variance array of x
    """

    x = np.array(x)
    x -= x.mean(axis=0)
    x /= x.std(axis=0)
    return x


def getEigen(covMatrix, reverse=True):
    r""" Get eigenvectors and values of hermitian matrix:

    Arguments:
        covMatrix (array-like):
            hermitian matrix
        reverse (bool):
            if True (default): order eigenvalues (and vectors) in
            decreasing order

    Returns:
        (tuple):
            tuple containing:

            - eigenvectors
            - eigenvalues
    """
    covMatrix = np.array(covMatrix)

    S, U = la.eigh(covMatrix)
    if reverse:
        S = S[::-1]
        U = U[:, ::-1]
    return (S, U)


def getVariance(eigenvalue):
    r"""
    Based on input eigenvalue computes cumulative sum and normalizes to overall
    sum to obtain variance explained

    Arguments:
        eigenvalue (array-like):
            eigenvalues
    Returns:
        (float):
            variance explained
    """

    v = np.array(eigenvalue).cumsum()
    v /= v.max()
    return (v)


def regularize(m, verbose=True):
    r"""
    Make matrix positive-semi definite by ensuring minimum eigenvalue >= 0:
    add absolute value of minimum eigenvalue and 1e-4 (for numerical stability
    of abs(min(eigenvalue) < 1e-4 to diagonal of matrix

    Arguments:
        m (array-like):
            symmetric matrix

    Returns:
       (tuple):
            Returns tuple containing:

            - positive, semi-definite matrix from input m (numpy array)
            - minimum eigenvalue of input m
    """
    S, U = la.eigh(m)
    minS = S.min()
    if minS < 0:
        verboseprint(
            "Regularizing: minimum Eigenvalue %6.4f" % S.min(),
            verbose=verbose)
        m += (abs(S.min()) + 1e-4) * sp.eye(m.shape[0])
    elif minS < 1e-4:
        verboseprint("Make numerically stable: minimum Eigenvalue %6.4f" %
                     S.min(), verbose=verbose)
        m += 1e-4 * sp.eye(m.shape[0])
    else:
        verboseprint("Minimum Eigenvalue %6.4f" % S.min(), verbose=verbose)

    return (m, minS)


def generate_permutation(P, S, n, seed=12321, exclude_zero=False):
    r"""
    Generate permutation.

    Arguments:
        seed (int):
            used as seed for pseudo-random numbers generation; default: 12321
        n (int):
            number of permutations to generated
        P (int):
            total number of traits
        S (int):
            subsampling size
        exclude_zero (bool):
            should zero be in set to draw from

    Returns:
        (list):
            Returns list of length n containing [np.arrays] of length [`S`]
            with subsets/permutations of numbers range(P)
    """
    rand_state = np.random.RandomState(seed)
    return_list = [None] * n
    if exclude_zero:
        rangeP = list(range(P))[1:]
    else:
        rangeP = list(range(P))
    for i in range(n):
        perm_dic = rand_state.choice(a=rangeP, size=S, replace=False)
        return_list[i] = perm_dic
    return return_list


def inflate_matrix(bootstrap_traits, bootstrap, P, zeros=True):
    r"""
    Project small matrix into large matrix using indeces provided:

    Arguments:
        bootstrap_traits (array-like):
            [`S` x `S`] covariance matrix estimates
        bootstrap (array-like):
            [`S` x 1] array with indices to project [`S` x `S`] matrix values
            into [`P` x `P`] matrix
        P (int):
            total number of dimensions
        zeros (bool):
            fill void spaces in large matrix with zeros (True,
            default) or nans (False)

    Returns:
        (numpy array):
            Returns [`P` x `P`] matrix containing [`S` x `S`] matrix values at
            bootstrap indeces and zeros/nans elswhere
    """
    index = np.ix_(np.array(bootstrap), np.array(bootstrap))
    if zeros is True:
        all_traits = np.zeros((P, P))
    else:
        all_traits = nans((P, P))
    all_traits[index] = bootstrap_traits
    return (all_traits)


def verboseprint(message, verbose=True):
    r"""
    Print message if verbose option is True.

    Arguments:
        message (string):
            text to print
        verbose (bool):
            flag whether to print message (True) or not (False)
    """
    if verbose is True:
        print(message)


def match(samples_ref, data_compare, samples_compare, squarematrix=False):
    r"""
    Match the order of data and ID matrices to a reference sample order,

    Arguments:
        samples_ref (array-like):
            [`M`] sammple Ids used as reference
        data_compare (array-like):
            [`N` x `L`] data matrix with [`N`] samples and [`L`] columns
        samples_compare (array-like):
            [`N`] sample IDs to be matched to samples_ref
        squarematrix (bool):
            is data_compare a square matrix i.e. samples in cols and rows

    Returns:
        (tuple):
            tuple containing:

            - data_compare (numpy array):
              [`M` x `L`] data matrix of input data_compare
            - samples_compare (numpy array):
              [`M`] sample IDs of input samples_compare
            - samples_before (int):
              number of samples in data_compare/samples_compare before matching
              to samples_ref
            - samples_after (int):
              number of samples in data_compare/samples_compare after matching
              to samples_ref
    """
    samples_before = samples_compare.shape[0]
    subset = pd.match(samples_ref, samples_compare)

    data_compare = data_compare[subset, :]
    if squarematrix:
        data_compare = data_compare[:, subset]
    samples_compare = samples_compare[subset]
    samples_after = samples_compare.shape[0]
    np.testing.assert_array_equal(
        samples_ref,
        samples_compare,
        err_msg=("Col order does not match. These"
                 "are the differing columns:\t%s") %
        (np.array_str(np.setdiff1d(samples_ref, samples_compare))))
    return (data_compare, samples_compare, samples_before, samples_after)


def AlleleFrequencies(snp):
    hc_snps = np.array([makeHardCalledGenotypes(s) for s in snp])
    counts = np.array(np.unique(hc_snps, return_counts=True))
    frequencies = counts[1, :] / float(len(hc_snps))
    major_a = sqrt(frequencies.max())
    minor_a = 1 - major_a
    return minor_a, major_a


def makeHardCalledGenotypes(snp):
    if snp <= 0.5:
        return 0
    elif snp > 1.5:
        return 2
    else:
        return 1


def effectiveTests(test):
    # 1. get correlation matrix
    corr_matrix = stats.spearmanr(test).correlation
    # 2. Get eigenvalues of correlation matrix:
    eigenval, eigenvec = la.eigh(corr_matrix)
    # 3. Determine effective number of tests:
    t = np.sqrt(eigenval).sum()**2 / eigenval.sum()
    return t

def find_square(i_coord, j_coord, side_length):
    return np.union1d(np.array(range(i_coord, i_coord+side_length)),
        np.array(range(j_coord, j_coord+side_length)))

# Relabel a subset so we don't need to recompute set covers for large m
def relabel_subset(subset, permutation):
    return [permutation[x] for x in subset]

########
# This method computes multiple set covers of tuples.
# To do so we consider instead covering a matrix is size
# number_of_traits*number_of_traits.
########

def multiple_set_covers_all(number_of_traits, sample_size, number_of_covers,
        seed=2152):
    sample_size_t = sample_size
    #We make odd sized samples even and then fix the sampling size later.
    if (sample_size % 2) != 0:
        sample_size_t = sample_size -1

    # Here we compute one set cover. After that we can generate the others based
    # on this by relabelling the matrix rows/cols with a permutation
    i = 0 
    j = 0
    used_subsets = list()
    while (j < number_of_traits):
        while (i < number_of_traits):
            # We're on the main diagonal and we can get a better set than normal.
            if i==j:
                used_subsets.append(find_square(i,j,sample_size_t) % number_of_traits)
                i += sample_size_t
            # We're not on the main diagonal so we can only cover a square of
            # size sample_size/2*sample_size/2
            else:
                used_subsets.append(find_square(i,j,sample_size_t//2) % number_of_traits)
                i += (sample_size_t//2)
        j += sample_size_t//2
        if j%sample_size_t ==0:
            i=j
        else:
            # Set i to new start position
            i = (j//sample_size_t)*sample_size_t +sample_size_t

    counts = sp.zeros((number_of_traits, number_of_traits))
    # Don't bother recomputing the set cover just relabel in randomly.
    # (you can do it in a non-random way but the coverage will look cluster
    # around the main diagonal
    if (sample_size % 2) !=0 :
        for i in range(0,len(used_subsets)):
            used_subsets[i] = np.append(used_subsets[i],
                    np.random.randint(0,number_of_traits))

    bootstrap_array = list()
    for num in range(0,number_of_covers):
        order = np.random.permutation(number_of_traits)
        for i in range(0,len(used_subsets)):
            bootstrap_array.append(relabel_subset(used_subsets[i], order))
            index = np.ix_(relabel_subset(used_subsets[i], order),
                relabel_subset(used_subsets[i], order))
            counts[index] += 1
    return {'bootstrap': bootstrap_array, 'counts': counts}
