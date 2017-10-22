######################
### import modules ###
######################


import sys
sys.path.append('./../../')

from limmbo.utils.utils import verboseprint
from limmbo.utils.utils import nans
from limmbo.utils.utils import regularize
from limmbo.utils.utils import inflate_matrix

#
import scipy as sp
from scipy.optimize import fmin_l_bfgs_b as opt
import pandas as pd
import numpy as np
import bottleneck as bn
import time
import cPickle
import matplotlib as mpl
mpl.use('Agg')

import limix.deprecated as dlimix
import mtSet
import mtSet.pycore.modules.multiTraitSetTest as MTST

import pp

######################
### core functions ###
#####################

class DataLimmbo(object):
    def __init__(self, datainput, options=None):
        '''
        nothing to initialize
        '''
        self.options = options
        self.phenotypes = datainput.phenotypes
        self.relatedness = datainput.relatedness
        self.nrsamples = None
        self.nrtraits = None

        self.Cg = None
        self.Cn = None
        self.processtime = None
        self.bootstrap = None

    def generateBootstrapMatrix(self, seed=12321, P=100, p=10,
                                minCooccurrence=3, n=None):
        """
        Generate subsampling matrix.
        Input:
            * seed: seed [int] for pseudo-random numbers generation; 
              default: 12321
            * P: total number [int] of traits
            * p: size [int] of permutation subset
            * minCooccurrence: minimum number [int] a trait pair
              should be sampled; once reached for all trait pairs, sampling is 
              stopped if n is None; default=3
            * n: if not None, sets the total number [int] of permutations, 
              otherwise n determined by minCooccurrence;  default: None
        Output:
            * self.runs: n of n not None, or determined once all trait-trait
              subsamplings have occurrd minCooccurence times
            * return_list: [list] of length self.runs containing [1 x S] 
              [np.array] with sample of numbers range(P)
            * counts_min: minimum trait-trait co-occurrence [int] in sampling 
              matrix
        """
        rand_state = np.random.RandomState(self.options.seed)
        counts = sp.zeros((self.options.P, self.options.P))
        return_list = []

        if n is not None:
            verboseprint(("Generate bootstrap matrix with %s bootstrap samples "
                    "(number of specified bootstraps") % n, 
                    verbose=self.options.verbose)
            for i in xrange(n):
                bootstrap = rand_state.choice(a=range(self.options.P),
                                              size=self.options.p,
                                              replace=False)
                return_list.append(bootstrap)
                index = np.ix_(np.array(bootstrap), np.array(bootstrap))
                counts[index] += 1
            self.runs = n
        else:
            while counts.min() < minCooccurrence:
                bootstrap = rand_state.choice(a=range(self.options.P),
                                              size=self.options.p,
                                              replace=False)
                return_list.append(bootstrap)
                index = np.ix_(np.array(bootstrap), np.array(bootstrap))
                counts[index] += 1
            self.runs = len(return_list)
            verboseprint(("Generated bootstrap matrix with %s bootstrap runs "
                        " such that each trait-trait was sampled %s") % 
                    (self.runs, minCooccurrence),
                    verbose=self.options.verbose)

        return return_list, counts.min()

    def bootstrapPhenotypes(self, bs, bootstrap_matrix):
        """ 
        Subsample [S] phenotypes with [N] samples form total of [P] 
        phenotypes. Indeces for subsampling provided in [bs x S] 
        bootstrap_matrix, where bs is the bs is the total number of bootstraps
        as determined by .generateBootstrapMatrix()
        Input:
            * bs: [scalar] bootstrap index
            * bootstrap_matrix: [bs x S] [pd.Dataframe] with subsampling 
                                indeces for self.phenotypes
        Output:
            * phenotypes: [N x S] [np.array] of subsampled phenotypes 
        """
        bootstrap = bootstrap_matrix.iloc[bs, :]
        phenotypes = self.phenotypes[:, bootstrap]
        return phenotypes

    
    def sampleCovarianceMatricesPP(self):
        """
        Distribute variance decomposition of subset matrices via pp
        Input:
            * self.runs: number [int] of bootstrapping runs executed for this 
              experiment
            * self.options.P: number [int] of phenotypes P
            * self.options.p:  subsampling size S [int], default: 10
            * self.options.output: output directory [string]; needed for 
            * self.options.seed: seed [int] to initialise random number
              generator for bootstrapping
            * minCooccurrence: minimum number [int] a trait pair
              should be sampled; once reached for all trait pairs, sampling is 
              stopped if n is None; default=3
            * n: if not None, sets the total number [int] of permutations, 
              otherwise n determined by minCooccurrence;  default: None
            * self.options.output: output directory [string]; needs writing
              permission
           * self.options.cpus: number [int] of cpus available for covariance
             estimation
           * self.options.verbose: [bool] should messages be printed to stdout
        Output:
            * results: results [list] of .VarianceDecomposition of the 
                       subset matrices
        """

        bootstrap_matrix, minimumTraitTraitcount = \
                self.generateBootstrapMatrix(seed=self.options.seed,
                        n=self.options.runs,
                        P=self.options.P, p=self.options.p,
                        minCooccurrence=self.options.minCooccurrence)
        bootstrap_matrix = pd.DataFrame(bootstrap_matrix)
        bootstrap_matrix.to_csv("%s/bootstrap_matrix.csv" %
                                self.options.output, sep=",",
                                index=True, header=False)
        ppservers = ()
        jobs = []
        results = []

        if self.options.cpus is not None:
            job_server = pp.Server(self.options.cpus, ppservers=ppservers)
        else:
            job_server = pp.Server(ppservers=ppservers)

        verboseprint("Number of CPUs available for parallelising: %s" % 
                    job_server.get_ncpus(),
                     verbose=self.options.verbose)
        
        self.nrtraits, self.nrsamples = self.phenotypes.shape
        outfile = "%s/mtSetresults_nrsamples%s_nrtraits%s.h5" % (
            self.options.output, self.nrsamples, self.nrtraits)
        for bs in range(self.runs):
            pheno = self.bootstrapPhenotypes(bs, bootstrap_matrix)
            verboseprint("Start vd for bootstrap nr %s" % bs)
            jobs.append(job_server.submit(self.VarianceDecomposition,
                (pheno, bs),
                (verboseprint,),
                ("mtSet", "time")))
        
        for job in jobs:
            bsresult = job()
            bsresult['bootstrap'] = bootstrap_matrix.iloc[
                    bsresult['bsindex'], :]
            results.append(bsresult)

        return results

    def combineBootstrap(self, results):
        """
        Combine the [S x S] subset covariance matrices to find the overall
        [P x P] covariance matrices Cg and Cn and write as .csv files
        Input:
            * results: results [list] of sampleCovarianceMatricesPP()
            * self.options.timing: [bool] should runtime be recorded and
              written to file
            * self.options.output: output directory [string]; needs writing
              permission
            * self.options.seed: seed [int] to initialise random number
              generator for bootstrapping
        Return:
            * self
        """
        verboseprint("Combine bootstrapping results...",
                     verbose=self.options.verbose)
        time0 = time.clock()
        Cg_fit, Cn_fit, Cg_average, Cn_average, process_time_bs, nr_bs = \
                self.getBootstrapResults(results=results,
                        timing=self.options.timing)
        time1 = time.clock()
        
        if self.options.timing is True:
            proc_time_combine_bs = time1 - time0
            proc_time_sum_ind_bs = np.array(process_time_bs).sum()
            pd.DataFrame(process_time_bs).to_csv("%s/process_time_bs.csv" %
                                                 (self.options.output),
                                                 sep=",", header=False,
                                                 index=False)
            pd.DataFrame([proc_time_combine_bs, proc_time_sum_ind_bs],
                         index=["Proctime combine BS",
                                "Proctime sum of individual BS"]).to_csv(
                "%s/process_time_summary.csv" % (self.options.output),
                sep=",", header=False, index=True)

        Cg_average, Cg_average_ev_min = regularize(Cg_average)
        Cn_average, Cn_average_ev_min = regularize(Cn_average)
        Cg_fit, Cg_fit_ev_min = regularize(Cg_fit)
        Cn_fit, Cn_fit_ev_min = regularize(Cn_fit)

        verboseprint("Generate output files", verbose=self.options.verbose)
        pd.DataFrame(Cg_average).to_csv("%s/Cg_average_seed%s.csv" %
                                        (self.options.output,
                                         self.options.seed),
                                        sep=",", header=False, index=False)
        pd.DataFrame(Cn_average).to_csv("%s/Cn_average_seed%s.csv" %
                                        (self.options.output,
                                         self.options.seed),
                                        sep=",", header=False, index=False)

        pd.DataFrame(Cg_fit).to_csv("%s/Cg_fit_seed%s.csv" %
                                    (self.options.output, self.options.seed),
                                    sep=",", header=False, index=False)
        pd.DataFrame(Cn_fit).to_csv("%s/Cn_fit_seed%s.csv" %
                                    (self.options.output, self.options.seed),
                                    sep=",", header=False, index=False)

        return self

    def VarianceDecomposition(self, phenoSubset, bs=None):
        """Compute variance decomposition of phenotypes into genetic and noise 
        covariance
        Input:
            * phenoSubset: [N x S] [phenotypes [np.array] for which variance
              decomposition should be computed
            * bs: number of subsample [int]
            * self.phenotypes: [N x P] original phenotypes [np.array] 
            * self.relatedness: [N x N] kinship/genetic relatedness [np.array] 
              used in estimation of genetic component
            * self.options.output: output directory [string]; needed for 
              caching 
            * self.options.cache: [bool] should mtSet results be cached to 
            * self.options.verbose: [bool] should messages be printed to stdout
        Output:
            * dictionary containing:
                * Cg: [S x S] genetic variance component [np.array]
                * Cn: [S x S] noise variance component [np.array]
                * process_time: cpu time [double] of variance decomposition
                * bsindex: number of subsample [int]
        """

        outfile = None
        if self.options.cache is True and self.options.output is None:
            sys.exit(("Output directory must be specified if caching is "
            "enabled"))
        if self.options.cache is False and self.options.output is not None:
            print ("Warning: Caching is disabled, despite having supplied an "
                   "output directory")
        if self.options.cache is True and self.options.output is not None:
            self.nrtraits, self.nrsamples = self.phenotypes.shape
            outfile = "%s/mtSetresults_nrsamples%s_nrtraits%s.h5" % (
                self.options.output, self.nrsamples, self.nrtraits)

        # time variance decomposition
        t0 = time.clock()
        mtset = mtSet.pycore.modules.multiTraitSetTest.MultiTraitSetTest(
                Y=phenoSubset, XX=self.relatedness)
        mtset_null_info = mtset.fitNull(
            cache=self.options.cache, fname=outfile, 
            n_times=self.options.iterations,
            rewrite=True)
        t1 = time.clock()
        processtime = t1 - t0

        if mtset_null_info['conv']:
            verboseprint("mtSet for bootstrap number %s converged" % bs, 
                        verbose=self.options.verbose)
            Cg = mtset_null_info['Cg']
            Cn = mtset_null_info['Cn']
        else:
            verboseprint("mtSet for bootstrap number %s did not converge" % 
                    bs, verbose=self.options.verbose)
            Cg = None
            Cn = None
        return {'Cg': Cg, 'Cn': Cn, 
                'process_time': processtime,
                'bsindex': bs}

    def getBootstrapResults(self, results):
        """
        Collect bootstrap results of [S x S] traits and combine all runs to 
        total [P x P] covariance matrix
        Input:
            * results: results [list] of sampleCovarianceMatricesPP()
            * self.runs: number [int] of bootstrapping runs executed for this 
              experiment
            * self.options.P: number [int] of phenotypes P
            * self.options.p:  subsampling size S [int], default: 10
            * self.options.output: output directory [string]; needed for 
              pickling all variance decomposition runs of [S x S] Cg and Cn
            * self.options.verbose: [bool] should messages be printed to stdout
        Output:
            * Cg_opt: [P x P] genetic covariance matrix via fitting
            * Cn_opt: [P x P] noise covariance matrix via fitting
            * Cg_norm: [P x P] genetic covariance matrix via simple average
            * Cn_norm: [P x P] noise covariance matrix via simple average
            * process_time_bs: [list] of run times for all variance 
              decomposition runs of [S x S] Cg and Cn
            * number_of_bs: total number [int] of successful bootstrapping runs
        """

        # list to contain trait indeces of each bootstrap run
        bootstrap = {}
        #sample_ID = None

        # create np.arrays of dimension PxP for mean and std of each entry in
        # covariance matrix
        Cg_norm = sp.zeros((self.options.P, self.options.P))
        Cn_norm = sp.zeros((self.options.P, self.options.P))

        # Process time per bootstrap
        process_time_bs = []

        # create np.arrays of dimension runs x P x P to store the bootstrapping
        # results for averaging
        Cg_average = nans((self.runs, self.options.P, self.options.P))
        Cn_average = nans((self.runs, self.options.P, self.options.P))

        Cg_fit = nans((self.runs, self.options.p, self.options.p))
        Cn_fit = nans((self.runs, self.options.p, self.options.p))

        n = 0
        #for vdresult in iter(resultsQ.get, "STOP"):
        for vdresult in results:
            bootstrap[n] = vdresult['bootstrap'].values
            if self.options.timing == True:
                process_time_bs.append(vdresult['process_time'])

            # store results of each bootstrap as matrix of inflated
            # matrices: NAs for traits that were not sampled
            if vdresult['Cg'] is None or vdresult['Cn'] is None:
                continue
            Cg_average[n, :, :] = inflate_matrix(
                vdresult['Cg'], bootstrap[n], P=self.options.P, zeros=False)
            Cn_average[n, :, :] = inflate_matrix(
                vdresult['Cn'], bootstrap[n], P=self.options.P, zeros=False)

            # store results of each bootstrap as matrix of small p x p
            # matrices generated by each bootstrap
            Cg_fit[n, :, :] = vdresult['Cg']
            Cn_fit[n, :, :] = vdresult['Cn']
            n += 1
        # Total number of successful bootstrapping runs
        number_of_bs = n - 1

        cPickle.dump(Cg_fit, open("%s/Cg_all_bootstraps.p" % 
            self.options.output, "wb"))
        cPickle.dump(Cn_fit, open("%s/Cn_all_bootstraps.p" % 
            self.options.output, "wb"))
        # Computing  mean and standard deviation of bootstrapping results at
        # each position p1, p2 in overall PxP covariance matrix
        verboseprint(("Computing mean of bootstrapping results"),
                     verbose=self.options.verbose)
        for p1 in range(self.options.P):
            for p2 in range(self.options.P):
                vals = Cg_average[:, p1, p2]
                Cg_norm[p1, p2] = vals[~sp.isnan(vals)].mean()
                vals = Cn_average[:, p1, p2]
                Cn_norm[p1, p2] = vals[~sp.isnan(vals)].mean()

        verboseprint(("Fitting bootstrapping results: minimize residual sum of"
                      "squares over all bootstraps"),
                     verbose=self.options.verbose)
        Cg_opt = self.fit_bootstrap_results(
            cov_init=Cg_norm, cov_bootstrap=Cg_fit,
            bootstrap_indeces=bootstrap,
            number_of_bs=number_of_bs, name="Cg")
        Cn_opt = self.fit_bootstrap_results(
            cov_init=Cn_norm, cov_bootstrap=Cn_fit,
            bootstrap_indeces=bootstrap,
            number_of_bs=number_of_bs,  name="Cn")

        return Cg_opt, Cn_opt, Cg_norm, Cn_norm, process_time_bs, number_of_bs

    def fit_bootstrap_results(self, cov_init, cov_bootstrap, bootstrap_indeces,
                              number_of_bs, name):
        """
        Fit  bootstrap results of [S x S] traits and combine all runs to total 
        [P x P] covariance matrix 
        Input:
            * lp: number [int] of phenotypes P
            * cov_init: [P x P] covariance matrix [np. array] used to 
              initialise fitting
            * number_of_bs: number [int] of successful bootstrapping runs 
              executed for this experiment
            * cov_bootstrap: [number_of_bs x S x S] bootstrap results 
              [np.array]
            * bootstrap_indeces: [number_of_bs x S] matrix of 
              bootstrapping indeces [np.array]
            * name: name [string] of covariance matrix
        Output:
            * C_opt_value: [P x P] covariance matrix [np.array] if fit 
              successful, else 1x1 matrix containing string: 'did not converge'
            * writes pickle output file with optimize settings and results to 
              directory
        """

        # initialise free-from Covariance Matrix: use simple average of
        # bootstraps as initial values
        # make use of Choleski decomposition and get parameters associated with
        # cov_init
        C_init = dlimix.CFreeFormCF(self.options.P)
        C_init.setParamsCovariance(cov_init)
        params = C_init.getParams()

        # initialise free-form covariance matrix C_fit: parameters to be set in
        # optimize function
        C_fit = dlimix.CFreeFormCF(self.options.P)

        # Fit bootstrap results to obtain closest covariance matrix
        # use parameters obtained from mean-initiated covariance matrix above
        verboseprint("Fitting parameters (minimizing rss via BFGS)...",
                     verbose=self.options.verbose)
        res = opt(self.rss, x0=params, args=(C_fit, number_of_bs,
                                             bootstrap_indeces, cov_bootstrap),
                  factr=1e12, iprint=2)
        cPickle.dump(res, open("%s/%s_opt_result.p" %
                               (self.options.output, name), "wb"))

        if res[2]['warnflag'] == 0:
            C_opt = dlimix.CFreeFormCF(self.options.P)
            C_opt.setParams(res[0])
            C_opt_value = C_opt.K()
        else:
            C_opt_value = np.array(['did not converge'])
        return(C_opt_value)

    def rss(self, params, C, n_bootstrap, index_bootstrap, list_C):
        """
        Compute residual sum of squares and gradient for estimate of covariance
        matrix C and bootstrapped values of C.
        Input:
            * params: np.array of parameters used for initialising estimate C 
              (length: 1/2*P*(P+1))
            * C: intialised free-form covariance matrix, to be fitted
            * n_bootstrap: number [int] of successful bootstrapping runs
              executed for this experiment
            * index_bootstrap: [number_of_bs x S] matrix of
              bootstrapping indeces [np.array]
            * list_C: list of n_bootstrap [p x p] bootstrapped covariance 
              matrices
        Output:
            * residual sum of squares
            * gradient of residual sum of squares function
        """

        # initialise values of covariance matrix with parameters (based on
        # cholesky decompostion)
        C.setParams(params)
        # get values of the covariance matrix
        C_value = C.K()
        # number of parameters: 1/2*P*(P+1)
        n_params = params.shape[0]
        # compute residual sum of squares between estimate of P x P Cg and
        # bootstrapped S x S Cgs
        RSS_res, index = self.RSS_compute(
            n_bootstrap, C_value, list_C, index_bootstrap)
        # compute gradient (first derivative of rss at each bootstrap)
        RSS_grad_res = self.RSS_grad_compute(
            n_bootstrap, n_params, C, C_value, list_C, index)

        return (RSS_res, RSS_grad_res)

    @staticmethod
    def RSS_compute(n_bootstrap,  C_value, list_C, index_bootstrap):
        """ 
        Compute residual sum of squares (rss) for each bootstrap run
        -  used parameter to be optimized  in quasi-Newton method (BGS) of 
           optimzation
        -  bootstrap matrices versus matrix to be optimized
        -  matrix to be optimized initialised with average over all bootstraps 
           for each position
        Input:
            * n_bootstrap: number [int] of successful bootstrap runs; 
            * C_value: [P x P] matrix [np.array] to be optimized
            * list_C: [n_bootstrap x p x p]  bootstrapped covariance 
              matrices [np.array] 
        Output:
            * res: added residual sum of squares [double] over all 
              n_bootstrap runs
            * index: [n_bootstrap x p] list; contains trait indeces used in 
              each bootstrap run; to be passed to gradient computation
        """
        res = 0
        index = {}
        for i in range(n_bootstrap):
            index[i] = np.ix_(index_bootstrap[i], index_bootstrap[i])
            res += bn.ss(C_value[index[i]] - list_C[i], axis=None)
        return (res, index)

    @staticmethod
    def RSS_grad_compute(n_bootstrap, n_params, C, C_value, list_C, index):
        """
        Compute gradient of residual sum of squares (rss) for each bootstrap 
        run at each index
        -  used as gradient parameter in quasi-Newton method (BFGS) of 
           optimzation
        -  matrix to be optimized initialised with average over all bootstraps 
           for each position
        Input:
            * n_bootstrap: number of [int] successful bootstrap runs;
            * n_params: number [int] of parameters of the model (parameters 
              needed to build positive, semi-definite matrix with Choleski 
              decomposition
            * C_value: [P x P] matrix [np.array] to be optimized
            * list_C: [n_bootstrap x p x p]  boostrapped covariance 
              matrices [np.array]
            * index: [n_bootstrap x p] trait indeces [list] used in
                          each bootstrap run
        Output:
            * res: [n_params x 1] sum of gradient over all bootstrap 
              runs for each parameter to be fitted [np.array]  
        """
        res = sp.zeros(n_params)
        for pi in range(n_params):
            Cgrad = C.Kgrad_param(pi)
            for i in range(n_bootstrap):
                res[pi] += (2 * Cgrad[index[i]] *
                            (C_value[index[i]] - list_C[i])).sum()
        return(res)
