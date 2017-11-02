r"""
VDSimple title
--------------

Document bla bla.


.. autoclass:: DataVD
    :members:
"""
######################
### import modules ###
######################

import sys

from limmbo.utils.utils import verboseprint
from limmbo.utils.utils import regularize
from limix.mtset import MTSet as MTST

import pandas as pd
import time

######################
### core functions ###
######################


class DataVD(object):
    r"""Class documentation title.

    Body.

    Parameters
    ----------
    phenotypes : numpy.array
        Descirption about phenotypes.

    Example
    -------

    .. doctest::

        >>> print(5)
        5
    """
    def __init__(self, datainput, options=None):
        self.options = options

        self.phenotypes = datainput.phenotypes
        self.relatedness = datainput.relatedness
        self.nrsamples = None
        self.nrtraits = None
        self.Cg = None
        self.Cn = None
        self.processtime = None

    def simpleVD(self):
	"""
	Wrapper for .VarianceDecomposition(), computing variance decomposition
 	and writing output files
	Input:
            * self.options.output: output directory [string] to write output
	      of variance decomposition: Cg, Cn and processtime; user needs
	      writing permission 
	Output: 
            * self.Cg: [P x P] genetic variance component [np.array] 
	      (numerically stable via regularize))
            * self.Cn: [P x P] noise variance component [np.array]
 	      (numerically stable via regularize))
	"""

        verboseprint(
            "Estimate covariance matrices based on standard REML",
            verbose=self.options.verbose)
        self.VarianceDecomposition()
        # ensure that matrices are true positive semi-definite matrices
        self.Cg, Cg_ev_min = regularize(self.Cg)
        self.Cn, Cn_ev_min = regularize(self.Cn)
        # save predicted covariance matrics
        try:
            pd.DataFrame(self.Cg).to_csv('{}/Cg_mtSet.csv'.format(
                self.options.output), sep=",", header=False, index=False)
            pd.DataFrame(self.Cn).to_csv('{}/Cn_mtSet.csv'.format(
                self.options.output), sep=",", header=False, index=False)

            pd.DataFrame([self.processtime]).to_csv('{}/process_time_mtSet.',
                '.csv'.format(self.options.output), sep=",", header=False,
                index=False)
        except:
            raise IOError('Cannot write to {}: check writing permissions',
                '{}'.format(self.options.output)
        return self

    def VarianceDecomposition(self):
        """Compute variance decomposition of phenotypes into genetic and noise
        covariance
        Input:
            * self.phenotypes: [N x P] phenotype matrix [np.array] for which
              variance decomposition should be computed
            * self.relatedness: [N x N] kinship/genetic relatedness matrix
              [np.array]
            * self.options.output: output directory [string]; needed for
              caching
            * self.options.cache: [bool] should mtSet results be cached to
            * self.options.verbose: [bool] should messages be printed to stdout
        Output:
            * self.Cg: [P x P] genetic variance component [np.array]
            * self.Cn: [P x P] noise variance component [np.array]
            * self.process_time: cpu time [double] of variance decomposition
        """

        outfile = None
        if self.options.cache is True and self.options.output is None:
            sys.exit(("Output directory must be specified if caching is "
                      "enabled"))
        if self.options.cache is False and self.options.output is not None:
            print("Warning: Caching is disabled, despite having supplied an "
                  "output directory")
        if self.options.cache is True and self.options.output is not None:
            self.nrtraits, self.nrsamples = self.phenotypes.shape
            outfile = "%s/mtSetresults_nrsamples%s_nrtraits%s.h5" % (
                self.options.output, self.nrsamples, self.nrtraits)

        # time variance decomposition
        t0 = time.clock()
        mtSet = MTST.MultiTraitSetTest(Y=self.phenotypes, XX=self.relatedness)
        mtSet_null_info = mtSet.fitNull(
            cache=self.options.cache,
            fname=outfile,
            n_times=self.options.iterations,
            rewrite=True)
        t1 = time.clock()

        if mtSet_null_info['conv']:
            verboseprint("mtSet converged", verbose=self.options.verbose)
            self.Cg = mtSet_null_info['Cg']
            self.Cn = mtSet_null_info['Cn']
        else:
            verboseprint(
                "mtSet did not converge", verbose=self.options.verbose)
        self.processtime = t1 - t0
