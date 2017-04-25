
# coding: utf-8
######################
### import modules ###
######################


import sys
sys.path.append('./../../')
#sys.path.append('/homes/hannah/bin/python_modules')
#sys.path.append('/homes/hannah/LiMMBo')
#sys.path.append(
   # '/nfs/gns/homes/hannah/software/python2.7.8/lib/python2.7/site-packages')

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

import limix.deprecated as dlimix
import mtSet.pycore.modules.multiTraitSetTest as MTST

from multiprocessing import Process, Queue, cpu_count

######################
### core functions ###
######################

class DataLimmbo(object):
    def __init__(self, datainput, options=None):
        '''
        nothing to initialize
        '''
        self.options = options

        self.phenotypes_pred = None
        self.phenotypes = datainput.phenotypes
        self.pheno_samples = datainput.pheno_samples
        self.phenotype_ID = datainput.phenotype_ID
        self.pheno_samples = datainput.pheno_samples

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
        Generate permutation.
        Input:
            * seed: numeric; used as seed for pseudo-random numbers generation; 
              default: 12321
            * P: numeric; total number of traits, default: 100
            * p: numeric; how small should the permutation subset be, 
              default: 10
            * minCooccurrence: numeric: minimum number of times a trait pair
              should be sampled; once reached for all trait pairs, sampling is 
              stopped if n is None; default=3
            * n: numeric; if not None, sets the total number of permutations, 
              otherwise n determined by minCooccurrence;  default: None
        Output;
            * return_list: list of length n containing np.array of length p 
              with permutation of numbers range(P)
            * minimum trait-trait co-occurrence in sampling matrix
        """
        rand_state = np.random.RandomState(self.options.seed)
        counts = sp.zeros((self.options.P, self.options.P))
        return_list = []

        if n is not None:
            for i in xrange(n):
                bootstrap = rand_state.choice(a=range(self.options.P),
                                              size=self.options.p,
                                              replace=False)
                return_list.append(bootstrap)
                index = np.ix_(np.array(bootstrap), np.array(bootstrap))
                counts[index] += 1
        else:
            while counts.min() < minCooccurrence:
                bootstrap = rand_state.choice(a=range(self.options.P),
                                              size=self.options.p,
                                              replace=False)
                return_list.append(bootstrap)
                index = np.ix_(np.array(bootstrap), np.array(bootstrap))
                counts[index] += 1

        return return_list, counts.min()

    def bootstrapPhenotypes(self, bs, bootstrap_matrix):
        # Get bootstrap indeces to be sampled form phenotype
        verboseprint("Bootstrap nr %s" % bs, verbose=self.options.verbose)
        bootstrap = bootstrap_matrix.iloc[bs, :]
        phenotype_ID = pd.DataFrame(np.array(self.phenotype_ID)[
                                    bootstrap.values].astype('str'),
                                    columns=["phenotype_ID"])
        phenotypes = self.phenotypes[:, bootstrap]
        return phenotypes

    def sampleCovarianceMatrices(self):

        def workerFunction(self, work_queue, done_queue, bsmat):
            self.nrtraits, self.nrsamples = self.phenotypes.shape
            outfile = "%s/mtSetresults_nrsamples%s_nrtraits%s.h5" % (
                self.options.output, self.nrsamples, self.nrtraits)
            for bs in iter(work_queue.get, 'STOP'):
                pheno = self.bootstrapPhenotypes(bs, bsmat)
                verboseprint("Start vd for bootstrap nr %s" % bs)
                Cg, Cn, proctime = self.VarianceDecomposition(
                    phenoSubset=pheno)
                done_queue.put({'Cg': Cg, 'Cn': Cn, 'process_time': proctime,
                                'bootstrap': bsmat.iloc[bs, :]})
            return True

        bootstrap_matrix, minimumTraitTraitcount = \
                self.generateBootstrapMatrix(seed=self.options.seed,
                        n=self.options.runs,
                        P=self.options.P, p=self.options.p,
                        minCooccurrence=self.options.minCooccurrence)
        bootstrap_matrix = pd.DataFrame(bootstrap_matrix)
        bootstrap_matrix.to_csv("%s/bootstrap_matrix.csv" %
                                self.options.output, sep=",",
                                index=True, header=False)

        workers = cpu_count()
        work_queue = Queue()
        done_queue = Queue()
        processes = []

        for bs in range(self.options.runs):
            work_queue.put(bs)

        for w in xrange(workers):
            p = Process(target=workerFunction, args=(
                self, work_queue, done_queue, bootstrap_matrix))
            p.start()
            processes.append(p)
            work_queue.put('STOP')

        for p in processes:
            p.join()

        done_queue.put('STOP')
        return done_queue

    def combineBootstrap(self, resultsQ):
        verboseprint("Combine bootstrapping results...",
                     verbose=self.options.verbose)
        time0 = time.clock()
        Cg_fit, Cn_fit, Cg_average, Cn_average, process_time_bs = \
                self.getBootstrapResults(resultsQ=resultsQ,
                        timing=self.options.timing)
        time1 = time.clock()

        Cg_average, Cg_average_ev_min = regularize(Cg_average)
        Cn_average, Cn_average_ev_min = regularize(Cn_average)

        verboseprint("Generate output files", verbose=self.options.verbose)

        pd.DataFrame(Cg_average).to_csv("%s/Cg_average_seed%s.csv" %
                                        (self.options.output,
                                         self.options.seed),
                                        sep=",", header=False, index=False)
        pd.DataFrame(Cn_average).to_csv("%s/Cn_average_seed%s.csv" %
                                        (self.options.output,
                                         self.options.seed),
                                        sep=",", header=False, index=False)

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

        Cg_fit, Cg_fit_ev_min = regularize(Cg_fit)
        Cn_fit, Cn_fit_ev_min = regularize(Cn_fit)
        pd.DataFrame(Cg_fit).to_csv("%s/Cg_fit_seed%s.csv" %
                                    (self.options.output, self.options.seed),
                                    sep=",", header=False, index=False)
        pd.DataFrame(Cn_fit).to_csv("%s/Cn_fit_seed%s.csv" %
                                    (self.options.output, self.options.seed),
                                    sep=",", header=False, index=False)

        return self

    def VarianceDecomposition(self, phenoSubset=None):
        """Compute variance decompostion of phenotypes into genetic and noise 
        covariance
        Input:
            * phenotypes: P x P np.array for which variance decomposition 
              should be computed
            * relatedness: np.array with kinship/genetic relatedness component 
              used in estimation of genetic component
            * outdir: string of output directory; needed for caching if 
              method =='mtSet'
            * seed: numeric; seed used to initialise mtSet, default=None
            * cache: bool; should mtSet results be cached to outdir, 
              default: True
        Output:
            * Cg: genetic variance component: P x P covariance np.array
            * Cn: noise variance component: P x P covariance np.array
            * processtime: cpu time of variance decomposition
        """

        outfile = None
        if self.options.cache is True and self.options.output is None:
            sys.exit(("Output directory must be specified if caching is"
            "enabled"))
        if self.options.cache is False and self.options.output is not None:
            print ("Warning: Caching is disabled, despite having supplied an"
                   "output directory")
        if self.options.cache is True and self.options.output is not None:
            self.nrtraits, self.nrsamples = self.phenotypes.shape
            outfile = "%s/mtSetresults_nrsamples%s_nrtraits%s.h5" % (
                self.options.output, self.nrsamples, self.nrtraits)

        if phenoSubset is None:
            pheno = self.phenotypes
        else:
            pheno = phenoSubset
        # time variance decomposition
        t0 = time.clock()
        mtSet = MTST.MultiTraitSetTest(Y=pheno, XX=self.relatedness)
        mtSet_null_info = mtSet.fitNull(
            cache=self.options.cache, fname=outfile, n_times=1000,
            rewrite=True, seed=self.options.seed)
        t1 = time.clock()

        if mtSet_null_info['conv']:
            verboseprint("mtSet converged", verbose=self.options.verbose)
            if phenoSubset is None:
                self.Cg = mtSet_null_info['Cg']
                self.Cn = mtSet_null_info['Cn']
                self.processtime = t1 - t0
            else:
                Cg = mtSet_null_info['Cg']
                Cn = mtSet_null_info['Cn']
                processtime = t1 - t0
        else:
            sys.exit("mtSet did not converge")

        if phenoSubset is None:
            return self
        else:
            return Cg, Cn, processtime

    def getBootstrapResults(self, resultsQ,  timing=True):
        """
        Collect bootstrap results of p x p traits and combine all runs to total 
        covariance matrix PxP
        Input:
            * directory: string, path to result files; top directory, may 
              contain subdirectories
            * resultsQ: 
            * runs: numeric, number of bootstrapping runs executed for this 
              experiment, default:10,000
            * P: numeric, PxP size of overall covariance matrix, default: 100
            * p: numierc, pxp size of small covariance matrx, default: 10
            * fit: bool, should estimate of Cg (average over all bootstraps) be 
              optimized with respect to residual sum of squares of estimate of 
              Cg and each bootstrap run, default:False
        Output:
            * Cg_norm_before:
            * Cn_norm_before:
            * Cg_norm_after:
            * Cn_norm_after:
            * bstrap:
            * sample_ID:
            * C_opt_value:
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
        Cg_average = nans((self.options.runs, self.options.P, self.options.P))
        Cn_average = nans((self.options.runs, self.options.P, self.options.P))

        Cg_fit = nans((self.options.runs, self.options.p, self.options.p))
        Cn_fit = nans((self.options.runs, self.options.p, self.options.p))

        n = 0
        for vdresult in iter(resultsQ.get, "STOP"):
            bootstrap[n] = vdresult['bootstrap'].values
            if self.options.timing == True:
                process_time_bs.append(vdresult['process_time'])

            # store results of each bootstrap as matrix of inflated
            # matrices: NAs for traits that were not sampled
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

        return Cg_opt, Cn_opt, Cg_norm, Cn_norm, process_time_bs

    def fit_bootstrap_results(self, cov_init, cov_bootstrap, bootstrap_indeces,
                              number_of_bs, name):
        """
        Fit  bootstrap results of p x p traits and combine all runs to total 
        covariance matrix PxP
        Input:
            * lp: numeric;  [P x P] is size of overall covariance matrix
            * cov_initial: [P x P] np.array;  covariance matrix used to 
              initialise fitting
            * cov_bootstrap: [number_of_bs x p x p ] np.array; matrix with 
              bootstrap results
            * number_of_bs: numeric; number of successful bootstrapping runs 
              executed for this experiment
            * bootstrap_indeces: [number_of_bs x p] np.array;  matrix of 
              bootstrapping indeces
        Output:
            * C_opt_value: P x P covariance matrix after fitting if fit 
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
            * n_bootstrap: numeric; how many bootstraps ran successfully
            * index_bootstrap: list of n_bootstrap bootstrap indeces of traits
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
        # compute residual sum of squares between estimate of PxP Cg and
        # bootstrapped pxp Cgs
        RSS_res, index = self.RSS_compute(
            n_bootstrap, C_value, list_C, index_bootstrap)
        # compute gradient (first derivative of rss at each bootstrap)
        RSS_grad_res = self.RSS_grad_compute(
            n_bootstrap, n_params, C, C_value, list_C, index)

        return (RSS_res, RSS_grad_res)

    @staticmethod
    # Functions to compute residual sum of squares and gradient of residual
    # sum of squares ###
    def RSS_compute(n_bootstrap,  C_value, list_C, index_bootstrap):
        """ 
        Compute residual sum of squares (rss) for each bootstrap run
        -  used parameter to be optimized  in quasi-Newton method (BGS) of 
           optimzation
        -  bootstrap matrices versus matrix to be optimized
        -  matrix to be optimized initialised with average over all bootstraps 
           for each position
        Input:
            * n_bootstrap: numeric; number of successful bootstrap runs; 
            * C_value: [P x P] np.array;  matrix to be optimized
            * list_C: [n_bootstrap x p x p] np.array; bootstrapped covariance 
              matrix 
        Output:
            * res: numeric;  added rss over all n_bootstrap runs
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
            * n_bootstrap: numeric; number of successful bootstrap runs;
            * n_params: numeric; number of parameters of the model (parameters 
              needed to build positive, semi-definite matrix with cholenski 
              decomposition
            * C_value:[P x P] np.array;  matrix to be optimized
            * list_C: [n_bootstrap x p x p] np.array; boostrapped covariance 
              matrix
        Output:
            * res: [n_params x 1] np.array; sum of gradient over all bootstrap 
              runs for each parameter to be fitted  
        """
        res = sp.zeros(n_params)
        for pi in range(n_params):
            Cgrad = C.Kgrad_param(pi)
            for i in range(n_bootstrap):
                res[pi] += (2 * Cgrad[index[i]] *
                            (C_value[index[i]] - list_C[i])).sum()
        return(res)
