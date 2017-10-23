######################
### import modules ###
######################

import sys
sys.path.append('./../../')

from limmbo.utils.utils import verboseprint
from limmbo.utils.utils import regularize
import mtSet.pycore.modules.multiTraitSetTest as MTST

import pandas as pd
import time



######################
### core functions ###
######################

class DataVD(object):
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
	Wrapper for .VarianceDecomposition(), copmuting variance decomposition
 	and writing output files
	Input:
            * self.options.output: output directory [string] to write output 
	      of variance decomposition: Cg, Cn and processtime; user needs
	      writing permission 
	Outout: 
            * self.Cg: [P x P] genetic variance component [np.array] 
	      (numerically stable via regularize))
            * self.Cn: [P x P] noise variance component [np.array]
 	      (numerically stable via regularize))
	"""

        verboseprint("Estimate covariance matrices based on standard REML", 
			verbose = self.options.verbose)
        self.VarianceDecomposition()
        # ensure that matrices are true positive semi-definite matrices
        self.Cg, Cg_ev_min = regularize(self.Cg)
        self.Cn, Cn_ev_min  = regularize(self.Cn)
        # save predicted covariance matrics
        pd.DataFrame(self.Cg).to_csv("%s/Cg_mtSet.csv" % self.options.output,
                                    sep=",", header=False, index=False)
        pd.DataFrame(self.Cn).to_csv("%s/Cn_mtSet.csv" % self.options.output, 
                                    sep=",", header=False, index=False)

        pd.DataFrame([self.processtime]).to_csv("%s/process_time_mtSet.csv" %
                                                 (self.options.output),
                                                 sep=",", header=False,
                                                 index=False)
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
            print ("Warning: Caching §is disabled, despite having supplied an "
                   "output directory")
        if self.options.cache is True and self.options.output is not None:
            self.nrtraits, self.nrsamples = self.phenotypes.shape
            outfile = "%s/mtSetresults_nrsamples%s_nrtraits%s.h5" % (
                self.options.output, self.nrsamples, self.nrtraits)

        # time variance decomposition
        t0 = time.clock()
        mtSet = MTST.MultiTraitSetTest(Y=self.phenotypes, XX=self.relatedness)
        mtSet_null_info = mtSet.fitNull(
            cache=self.options.cache, fname=outfile, 
            n_times=self.options.iterations,
            rewrite=True)
        t1 = time.clock()

        if mtSet_null_info['conv']:
            verboseprint("mtSet converged", verbose=self.options.verbose)
            self.Cg = mtSet_null_info['Cg']
            self.Cn = mtSet_null_info['Cn']
        else:
            verboseprint("mtSet did not converge", 
		verbose=self.options.verbose)
	self.processtime = t1 - t0

