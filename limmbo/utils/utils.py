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
def multiple_set_covers_all(number_of_traits, sample_size_ori, number_of_covers,
        seed=2152):
    #Compute a set of subsets that represent a multiple set covers
    number_to_tuple = dict()
    global_counter = 0
    
    if (number_of_covers % 2)==0:
        number_of_covers = number_of_covers//2
    else:
        number_of_covers = number_of_covers//2 +1
    
    number_of_trait_tuples = (number_of_traits*(number_of_traits-1))//2
    order = np.random.permutation(number_of_trait_tuples)
    
    #Create a dictionary to store mappings between tuples and traits. Should be a function really
    for i in range(0, number_of_traits):
        for j in range(i+1, number_of_traits):
            number_to_tuple[order[global_counter]] = (i,j)
            global_counter = global_counter +1
    
    
    inflated_sample_size = sample_size_ori-1
    
    goal_set_cover_size = number_of_trait_tuples//inflated_sample_size +1
    used_subsets = list()
    
    set_cover = range(0, inflated_sample_size)
    #Compute the set covers one at a time
    for i in range(1 ,number_of_covers+1):
        for j in range(1, goal_set_cover_size+1):
            new_set_cover = [((x+(i-1)+(j-1)*inflated_sample_size) % number_of_trait_tuples) for x in set_cover]
            used_subsets.append(np.array(new_set_cover))
    
    flattened_list = list()
    #Map back from tuple index to trait ids
    count=1
    for set_tuple in used_subsets:
        list_to_flat = [number_to_tuple[x] for x in set_tuple]
        flattened_list.append(np.array([item for sublist in list_to_flat for item in sublist]))
        count = count + 1
    #Remove duplicates from each subset
    forward_array   = list([np.unique(xi) for xi in flattened_list])
    
    bootstrap_array = list()
    bootstrap_array.extend(forward_array)
    
    length_array = [len(x) for x in bootstrap_array]
    
    maxlen = max(length_array)
    
    #Because of the mapping from traits to tuples the method might not always pick each set to be
    #the same size. It seemed like that was needed by the method so the below code randomly fixes 
    #the unequal set size. If the number of tuples is close to the sample size this will be slow
    
    rand_state = np.random.RandomState(seed)
    
    for i in range(0,len(bootstrap_array)):
        while (len(bootstrap_array[i])!=maxlen):
            bootstrap_array[i] = np.unique(np.append(rand_state.choice(
                    a=list(range(number_of_traits)),
                    size=(maxlen-len(bootstrap_array[i])),
                    replace=False),
                bootstrap_array[i]))
    return bootstrap_array
