from limmbo.utils.utils import verboseprint
from limmbo.utils.utils import nans
from limmbo.utils.utils import regularize
from limmbo.utils.utils import inflate_matrix

import scipy as sp
from scipy.optimize import fmin_l_bfgs_b as opt
import pandas as pd
import numpy as np
import bottleneck as bn
import time
import cPickle

import limix.mtset

from limix_core.covar import FreeFormCov
import pp


class DataMismatch(Exception):
    r"""Raised when dimensions of sample/ID names do not match dimension of
    corresponding data"""
    pass


class LiMMBo(object):
    r"""
    Class for variance decomposition.

    Arguments:
        datainput (:class:`limmbo.io.InputData`):
           Object containing relevant data for variance decomposition,
           at least phenotypes and relatedness matrix.
        timing (bool):
           if set to True, process time will be recorded
        iterations (int):
            number of iterations for paramter estimation steps
        S (int):
            subsampling size `S`
        verbose (bool):
            Set to true to print progress messages.

    """
    def __init__(self, datainput, S, timing=False, iterations=10,
                 verbose=False):
        self.phenotypes = datainput.phenotypes
        self.relatedness = datainput.relatedness
        self.S = S
        self.timing = timing
        self.iterations = iterations
        self.verbose = verbose

        try:
            self.phenotypes = np.array(self.phenotypes)
        except:
            raise IOError(
                "datainput.phenotypes cannot be converted to np.array")

        try:
            self.relatedness = np.array(self.relatedness)
        except:
            raise IOError(
                "datainput.relatedness cannot be converted to np.array")

        if self.S > self.phenotypes.shape[1]:
            raise DataMismatch(("Subsampling size S ({}) greater than number "
                                "of phenotypes ({})").format(
                                    self.S,
                                    self.phenotypes.shape[1]))

    def runBootstrapCovarianceEstimation(self, seed, cpus, minCooccurrence=3,
                                         n=None):
        r"""
        Distribute variance decomposition of subset matrices via pp

        Arguments:
            seed (int):
                seed to initialise random number generator for bootstrapping
            minCooccurrence (int):
                minimum number a trait pair should be sampled; once reached
                for all trait pairs, sampling is stopped if n is None;
                default=3
            n (int):
                if not None, sets the total number of permutations,
                otherwise n determined by minCooccurrence;  default: None
            cpus (int):
                number of cpus available for covariance estimation

        Returns:
            (list):
                list containing variance components for all bootstrap runs
        """

        self.P = self.phenotypes.shape[1]
        self.__generateBootstrapMatrix(seed=seed, n=n,
                                       minCooccurrence=minCooccurrence)
        ppservers = ()
        jobs = []
        results = []

        if cpus is not None:
            job_server = pp.Server(cpus, ppservers=ppservers)
        else:
            job_server = pp.Server(ppservers=ppservers)

        verboseprint(
            'Number of CPUs available for parallelising: {}'.format(
                job_server.get_ncpus()),
            verbose=self.verbose)

        for bs in range(self.runs):
            pheno = self.__bootstrapPhenotypes(bs)
            verboseprint('Start vd for bootstrap nr {}'.format(bs + 1))
            jobs.append(
                job_server.submit(self.__VarianceDecomposition, (pheno, bs),
                                  (verboseprint, ), ("limix.mtset", "time")))

        for job in jobs:
            bsresult = job()
            bsresult['bootstrap'] = self.bootstrap_matrix[bsresult[
                'bsindex'], :]
            results.append(bsresult)

        return results

    def combineBootstrap(self, results):
        r"""
        Combine the [`S` x `S`] subset covariance matrices to find the overall
        [`P` x `P`] covariance matrices Cg and Cn.

        Arguments:
            results (list):
                results of runBootstrapVarianceDecomposition()

        Returns:
            (dictionary):
                dictionary containing:

                - **Cg_fit** (numpy.array):
                  [`P` x `P`] genetic covariance matrix via fitting
                - **Cn_fit** (numpy.array):
                  [`P` x `P`] noise covariance matrix via fitting
                - **Cg_average** (numpy.array):
                  [`P` x `P`] genetic covariance matrix via simple average
                - **Cn_average** (numpy.array):
                  [`P` x `P`] noise covariance matrix via simple average
                - **Cg_all_bs** (numpy.array):
                  [`runs` x `S` x `S`] genetic covariance matrices of `runs`
                  phenotype subsets of size `S`
                - **Cn_all_bs** (numpy.array):
                  [`runs` x `S` x `S`] noise covariance matrices of `runs`
                  phenotype subsets of size `S`
                - **proc_time_ind_bs** (list):
                  individual run times for all variance decomposition runs of
                  [`S` x `S`] Cg and Cn
                - **proc_time_sum_ind_bs** (list):
                  sum of individual run times for all variance decomposition
                  runs of [`S` x `S`] Cg and Cn
                - **proc_time_combine_bs** (list):
                  run time for finding [`P` x `P`] trait covariance estimates
                  from fitting [`S` x `S`] bootstrap covariance estimates
                - **nr_of_bs** (int):
                  number of bootstrap runs
                - **nr_of_successful_bs** (int):
                  total number of successful bootstrapping runs i.e. variance
                  decomposition converged
                - **results_fit_Cg** ():
                  results parameters of the bfgs-fit of the genetic covariance
                  matrices (via scipy.optimize.fmin_l_bfgs_g)
                - **results_fit_Cn** ():
                  results parameters of the bfgs-fit of the noise covariance
                  matrices (via scipy.optimize.fmin_l_bfgs_g)
        """

        verboseprint('Combine bootstrapping results...', verbose=self.verbose)
        time0 = time.clock()
        bs_results = self.__getBootstrapResults(results=results)
        time1 = time.clock()

        proc_time_combine_bs = time1 - time0
        proc_time_sum_ind_bs = np.array(bs_results['process_time_bs']).sum()

        verboseprint("Check Cg (average):", verbose=self.verbose)
        Cg_average, Cg_average_ev_min = regularize(bs_results['Cg_average'])
        verboseprint("Check Cn (average):", verbose=self.verbose)
        Cn_average, Cn_average_ev_min = regularize(bs_results['Cn_average'])
        verboseprint("Check Cg (fit):", verbose=self.verbose)
        Cg_fit, Cg_fit_ev_min = regularize(bs_results['Cg_fit'])
        verboseprint("Check Cn (fit):", verbose=self.verbose)
        Cn_fit, Cn_fit_ev_min = regularize(bs_results['Cn_fit'])

        results = {'Cg_fit': Cg_fit, 'Cn_fit': Cn_fit,
                   'Cg_average': Cg_average, 'Cn_average': Cn_average,
                   'Cg_all_bs': bs_results['Cg_bs'],
                   'Cn_all_bs': bs_results['Cn_bs'],
                   'proc_time_ind_bs': bs_results['process_time_bs'],
                   'proc_time_sum_ind_bs': proc_time_sum_ind_bs,
                   'proc_time_combine_bs': proc_time_combine_bs,
                   'nr_bs': self.runs,
                   'nr_successful_bs': bs_results['number_of_successful_bs'],
                   'results_fit_Cg': bs_results['results_fit_Cg'],
                   'results_fit_Cn': bs_results['results_fit_Cn']
                   }

        return results

    def saveVarianceComponents(self, resultsCombineBootstrap, output,
                               intermediate=True):
        r"""
        Save variance components as comma-separated files or python objects
        (via Cpickle).

        Arguments:
            resultsCombineBootstrap (dictionary):

                - **Cg_fit** (numpy.array):
                  [`P` x `P`] genetic covariance matrix via fitting
                - **Cn_fit** (numpy.array):
                  [`P` x `P`] noise covariance matrix via fitting
                - **Cg_average** (numpy.array):
                  [`P` x `P`] genetic covariance matrix via simple average
                - **Cn_average** (numpy.array):
                  [`P` x `P`] noise covariance matrix via simple average
                - **Cg_all_bs** (numpy.array):
                  [`runs` x `S` x `S`] genetic covariance matrices of `runs`
                  phenotype subsets of size `S`
                - **Cn_all_bs** (numpy.array):
                  [`runs` x `S` x `S`] noise covariance matrices of `runs`
                  phenotype subsets of size `S`
                - **proc_time_ind_bs** (list):
                  individual run times for all variance decomposition runs of
                  [`S` x `S`] Cg and Cn
                - **proc_time_sum_ind_bs** (list):
                  sum of individual run times for all variance decomposition
                  runs of [`S` x `S`] Cg and Cn
                - **proc_time_combine_bs** (list):
                  run time for finding [`P` x `P`] trait covariance estimates
                  from fitting [`S` x `S`] bootstrap covariance estimates
                - **nr_of_bs** (int):
                  number of bootstrap runs
                - **nr_of_successful_bs** (int):
                  total number of successful bootstrapping runs i.e. variance
                  decomposition converged
                - **results_fit_Cg** ():
                  results parameters of the bfgs-fit of the genetic covariance
                  matrices (via scipy.optimize.fmin_l_bfgs_g)
                - **results_fit_Cn** ():
                  results parameters of the bfgs-fit of the noise covariance
                  matrices (via scipy.optimize.fmin_l_bfgs_g)

            output (string):
                path/to/directory where variance components will be saved;
                needs writing permission
            intermediate (bool):
                if set to True, intermediate variance components (average
                covariance matrices, bootstrap matrices and results parameter
                of BFGS fit) are saved

        Returns:
            None
        """

        verboseprint("Generate output files", verbose=self.verbose)

        verboseprint("Write [`P` x `P`] covariance matrices",
                     verbose=self.verbose)
        pd.DataFrame(resultsCombineBootstrap['Cg_fit']).to_csv(
            '{}/Cg_fit_seed{}.csv'.format(output, self.seed), sep=",",
            header=False, index=False)
        pd.DataFrame(resultsCombineBootstrap['Cn_fit']).to_csv(
            '{}/Cn_fit_seed{}.csv'.format(output, self.seed), sep=",",
            header=False, index=False)

        if intermediate:
            verboseprint("Write bootstrap matrix", verbose=self.verbose)
            pd.DataFrame(self.bootstrap_matrix).to_csv(
                '{}/bootstrap_matrix.csv'.format(output), sep=",", index=True,
                header=False)

            verboseprint("Save intermediate variance components",
                         verbose=self.verbose)
            verboseprint(("Write covariance matrices based on average of "
                          "bootstrap matrices"), verbose=self.verbose)
            pd.DataFrame(resultsCombineBootstrap['Cg_average']).to_csv(
                "%s/Cg_average_seed%s.csv" % (output, self.seed),
                sep=",", header=False, index=False)
            pd.DataFrame(resultsCombineBootstrap['Cn_average']).to_csv(
                "%s/Cn_average_seed%s.csv" % (output, self.seed),
                sep=",", header=False, index=False)

            verboseprint(("Pickle array of all [`S` x `S`] bootstrap "
                          "covariance matrices"), verbose=self.verbose)
            cPickle.dump(resultsCombineBootstrap['Cg_all_bs'],
                         open('{}/Cg_all_bootstraps.p'.format(output), "wb"))
            cPickle.dump(resultsCombineBootstrap['Cn_all_bs'],
                         open('{}/Cn_all_bootstraps.p'.format(output), "wb"))

            verboseprint(("Pickle result parameters of BFGS fit for fitting "
                          "the [`S` x `S`] bootstrap covariance matrices to "
                          "the [`P` x `P`] overall trait covariance matrices"),
                         verbose=self.verbose)
            cPickle.dump(resultsCombineBootstrap['results_fit_Cg'],
                         open("%s/optimise_results_Cg.p" % (output), "wb"))
            cPickle.dump(resultsCombineBootstrap['results_fit_Cg'],
                         open("%s/optimise_results_Cn.p" % (output), "wb"))

        if self.timing:
            verboseprint("Save process times", verbose=self.verbose)
            pd.DataFrame(resultsCombineBootstrap['proc_time_ind_bs']).to_csv(
                "%s/process_time_all_bootstraps.csv" % (output),
                sep=",",
                header=False,
                index=False)
            overall_time = pd.DataFrame(
                [resultsCombineBootstrap['proc_time_combine_bs'],
                 resultsCombineBootstrap['proc_time_sum_ind_bs']],
                index=["Proctime combine BS",
                       "Proctime sum of individual BS"])
            overall_time.to_csv("%s/process_time_summary.csv" % output,
                                sep=",", header=False, index=True)

    def __generateBootstrapMatrix(self, seed=12321, minCooccurrence=3,
                                  n=None):
        r"""
        Generate subsampling matrix.

        Arguments:
            seed (int, optional):
                for pseudo-random numbers generation; default: 12321
            minCooccurrence (int):
                minimum number a trait pair should be sampled; once reached
                for all trait pairs, sampling is stopped if n is None;
                default=3
            n (int):
                if not None, sets the total number of permutations,
                otherwise n determined by minCooccurrence;  default: None

        Returns:
            None:
                updates LiMMBo instance with:

             - **seed** (int):
               seed for pseudo-random numbers generation
             - **runs** (int):
                n, if n was not None, or determined once all trait-trait
                subsamplings have occurrd minCooccurence times
             - **counts_min** (int):
               minimum trait-trait co-occurrence in sampling matrix
             - **bootstrap_matrix** (numpy.array):
                [`runs` x `S`] matrix containing bootstrap samples of numbers
                range(`P`)
        """
        rand_state = np.random.RandomState(seed)
        counts = sp.zeros((self.P, self.P))
        return_list = []

        if n is not None:
            verboseprint(
                ('Generate bootstrap matrix with {} bootstrap samples '
                 '(number of specified bootstraps').format(n),
                verbose=self.verbose)
            for i in xrange(n):
                bootstrap = rand_state.choice(a=range(self.P), size=self.S,
                                              replace=False)
                return_list.append(bootstrap)
                index = np.ix_(np.array(bootstrap), np.array(bootstrap))
                counts[index] += 1
            self.runs = n
        else:
            while counts.min() < minCooccurrence:
                bootstrap = rand_state.choice(a=range(self.P), size=self.S,
                                              replace=False)
                return_list.append(bootstrap)
                index = np.ix_(np.array(bootstrap), np.array(bootstrap))
                counts[index] += 1
            self.runs = len(return_list)
            verboseprint(
                ('Generated bootstrap matrix with {} bootstrap runs '
                 'such that each trait-trait combination was '
                 'sampled {}').format(self.runs, minCooccurrence),
                verbose=self.verbose)

        self.seed = seed
        self.bootstrap_matrix = np.array(return_list)
        self.counts_min = int(counts.min())

    def __bootstrapPhenotypes(self, bs):
        r"""
        Subsample [`S`] phenotypes with [`N`] samples form total of [`P`]
        phenotypes. Indices for subsampling provided in [`bs` x `S`]
        LiMMBo.bootstrap_matrix, where `bs` is the total number of bootstraps

        Arguments:
            bs (int):
                bootstrap index

        Uses:
            self.bootstrap_matrix (array-like):
                [`bs` x `S`] with subsampling indeces for phenotypes

        Returns:
            (numpy.array):

                - **phenotypes**:
                  [`N` x `S`] of subsampled phenotypes
        """
        bootstrap = self.bootstrap_matrix[bs, :]
        phenotypes = self.phenotypes[:, bootstrap]

        return phenotypes

    def __getBootstrapResults(self, results):
        r"""
        Collect bootstrap results of [`S` x `S`] traits and combine all runs to
        total [`P` x `P`] covariance matrix

        Arguments:
            results (list):
                results of runBootstrapVarianceDecomposition()

        Uses:
            runs (int):
                number of bootstrapping runs executed for this experiment

        Returns:
            (dictionary):
                dictionary containing:

                - **Cg_fit** (numpy.array):
                  [`P` x `P`] genetic covariance matrix via fitting
                - **Cn_fit** (numpy.array):
                  [`P` x `P`] noise covariance matrix via fitting
                - **Cg_average** (numpy.array):
                  [`P` x `P`] genetic covariance matrix via simple average
                - **Cn_average** (numpy.array):
                  [`P` x `P`] noise covariance matrix via simple average
                - **Cg_bs** (numpy.array):
                  [`runs` x `S` x `S`] genetic covariance matrices of `runs`
                  phenotype subsets of size `S`
                - **Cn_bs** (numpy.array):
                  [`runs` x `S` x `S`] noise covariance matrices of `runs`
                  phenotype subsets of size `S`
                - **process_time_bs** (list):
                  run times for all variance decomposition runs of [`S` x `S`]
                  Cg and Cn
                - **number_of_successful_bs** (int):
                  total number of successful bootstrapping runs i.e. variance
                  decomposition converged
                - **results_fit_Cg** ():
                  results parameters of the bfgs-fit of the genetic covariance
                  matrices (via scipy.optimize.fmin_l_bfgs_g)
                - **results_fit_Cn** ():
                  results parameters of the bfgs-fit of the noise covariance
                  matrices (via scipy.optimize.fmin_l_bfgs_g)
        """

        # list to contain trait indices of each bootstrap run
        bootstrap = {}

        # create np.arrays of dimension PxP for mean and std of each entry in
        # covariance matrix
        Cg_average = sp.zeros((self.P, self.P))
        Cn_average = sp.zeros((self.P, self.P))

        # Process time per bootstrap
        process_time_bs = []

        # create np.arrays of dimension runs x P x P to store the bootstrapping
        # results for averaging
        Cg_bs_large = nans((self.runs, self.P, self.P))
        Cn_bs_large = nans((self.runs, self.P, self.P))

        Cg_bs = nans((self.runs, self.S, self.S))
        Cn_bs = nans((self.runs, self.S, self.S))

        n = 0
        for vdresult in results:
            bootstrap[n] = vdresult['bootstrap']
            process_time_bs.append(vdresult['process_time'])

            # store results of each bootstrap as matrix of inflated
            # matrices: NAs for traits that were not sampled
            if vdresult['Cg'] is None or vdresult['Cn'] is None:
                continue
            Cg_bs_large[n, :, :] = inflate_matrix(
                vdresult['Cg'], bootstrap[n], P=self.P, zeros=False)
            Cn_bs_large[n, :, :] = inflate_matrix(
                vdresult['Cn'], bootstrap[n], P=self.P, zeros=False)

            # store results of each bootstrap as matrix of small S x S
            # matrices generated by each bootstrap
            Cg_bs[n, :, :] = vdresult['Cg']
            Cn_bs[n, :, :] = vdresult['Cn']
            n += 1
        # Total number of successful bootstrapping runs
        number_of_bs = n - 1

        # Computing mean of bootstrapping results at
        # each position p1, p2 in overall PxP covariance matrix
        verboseprint(
            ("Computing mean of bootstrapping results"),
            verbose=self.verbose)
        for p1 in range(self.P):
            for p2 in range(self.P):
                vals = Cg_bs_large[:, p1, p2]
                Cg_average[p1, p2] = vals[~sp.isnan(vals)].mean()
                vals = Cn_bs_large[:, p1, p2]
                Cn_average[p1, p2] = vals[~sp.isnan(vals)].mean()

        Cg_reg, ev_g = regularize(Cg_average)
        Cn_reg, ev_n,  = regularize(Cn_average)

        verboseprint(
            ("Fitting bootstrapping results: minimize residual sum of "
             "squares over all bootstraps"),
            verbose=self.verbose)
        Cg_fit, results_fit_Cg = self.__fit_bootstrap_results(
            cov_init=Cg_reg,
            cov_bootstrap=Cg_bs,
            bootstrap_indeces=bootstrap,
            number_of_bs=number_of_bs,
            name="Cg")
        Cn_fit, results_fit_Cn = self.__fit_bootstrap_results(
            cov_init=Cn_reg,
            cov_bootstrap=Cn_bs,
            bootstrap_indeces=bootstrap,
            number_of_bs=number_of_bs,
            name="Cn")

        results = {'Cg_bs': Cg_bs, 'Cn_bs': Cn_bs,
                   'Cg_fit': Cg_fit, 'Cn_fit': Cn_fit,
                   'Cg_average': Cg_average, 'Cn_average': Cn_average,
                   'process_time_bs': process_time_bs,
                   'number_of_successful_bs': number_of_bs,
                   'results_fit_Cg': results_fit_Cg,
                   'results_fit_Cn': results_fit_Cn,
                   }

        return results

    def __fit_bootstrap_results(self, cov_init, cov_bootstrap,
                                bootstrap_indeces, number_of_bs, name):
        r"""
        Fit  bootstrap results of [`S` x `S`] traits and combine all runs to
        total [`P` x `P`] covariance matrix.

        Arguments:
            cov_init (array-like):
                [`P` x `P`] covariance matrix used to initialise fitting
            number_of_bs (int):
                number of successful bootstrapping runs executed for this
                experiment
            cov_bootstrap (array-like):
                [`number_of_bs` x `S` x `S`] bootstrap results
            bootstrap_indeces (array-like):
                [`number_of_bs` x `S`] matrix of bootstrapping indeces
            name (string):
                name of covariance matrix

        Returns:
            (tuple):
                tuple containing:

                - **C_opt_value** (array-like):
                  [`P` x `P`] covariance matrix if fit successful, else 1x1
                  matrix containing string 'did not converge'
                - **results_fit** ():
                  results parameters of the bfgs-fit of the covariance
                  matrix (via scipy.optimize.fmin_l_bfgs_g)


        """

        # initialise free-from Covariance Matrix: use simple average of
        # bootstraps as initial values
        # make use of Choleski decomposition and get parameters associated with
        # cov_init
        C_init = FreeFormCov(self.P)
        C_init.setCovariance(cov_init)
        params = C_init.getParams()

        # initialise free-form covariance matrix C_fit: parameters to be set in
        # optimize function
        C_fit = FreeFormCov(self.P)

        # Fit bootstrap results to obtain closest covariance matrix
        # use parameters obtained from mean-initiated covariance matrix above
        verboseprint(
            "Fitting parameters (minimizing rss via BFGS)...",
            verbose=self.verbose)
        results_fit = opt(
            self.__rss,
            x0=params,
            args=(C_fit, number_of_bs, bootstrap_indeces, cov_bootstrap),
            factr=1e12,
            iprint=2)

        if results_fit[2]['warnflag'] == 0:
            C_opt = FreeFormCov(self.P)
            C_opt.setParams(results_fit[0])
            C_opt_value = C_opt.K()
        else:
            C_opt_value = np.array(['did not converge'])

        return C_opt_value, results_fit

    def __rss(self, params, C, n_bootstrap, index_bootstrap, list_C):
        r"""
        Compute residual sum of squares and gradient for estimate of covariance
        matrix C and bootstrapped values of C.

        Arguments:
            params (array-like):
                of parameters used for initialising estimate C
                (length: 1/2*`P`*(`P`+1))
            C:
                intialised free-form covariance matrix, to be fitted
            n_bootstrap (int):
                number of successful bootstrapping runs executed for this
                experiment
            index_bootstrap (numpy.array):
                [`number_of_bs` x `S`] matrix of bootstrapping indeces
            list_C (list):
                list of n_bootstrap [`S` x `S`] bootstrapped covariance
                matrices

        Returns:
            (tuple):
                tuple containing:

                - **RSS_res**:
                  residual sum of squares
                - **RSS_grad_res**:
                  gradient of residual sum of squares function

        """

        # initialise values of covariance matrix with parameters (based on
        # cholesky decomposition)
        params = np.array(params)
        C.setParams(params)
        # get values of the covariance matrix
        C_value = C.K()
        # number of parameters: 1/2*P*(P+1)
        n_params = params.shape[0]
        # compute residual sum of squares between estimate of P x P Cg and
        # bootstrapped S x S Cgs
        RSS_res, index = self.__RSS_compute(n_bootstrap, C_value, list_C,
                                            index_bootstrap)
        # compute gradient (first derivative of rss at each bootstrap)
        RSS_grad_res = self.__RSS_grad_compute(n_bootstrap, n_params, C,
                                               C_value, list_C, index)

        return RSS_res, RSS_grad_res

    @staticmethod
    def __RSS_compute(n_bootstrap, C_value, list_C, index_bootstrap):
        r"""
        Compute residual sum of squares (rss) for each bootstrap run

        - used parameter to be optimized in quasi-Newton method (BFGS) of
          optimization
        - bootstrap matrices versus matrix to be optimized
        - matrix to be optimized initialised with average over all bootstraps
          for each position

        Arguments:
            n_bootstrap (int):
                number of successful bootstrap runs;
            C_value (array-like):
                [`P` x `P`] matrix to be optimized
            list_C (array-like):
                [`n_bootstrap` x `S` x `S`]  bootstrapped covariance matrices

        Returns:
            (tuple):
                tuple containing:

                - **res** (double):
                  sum of residual sum of squares over all `n_bootstrap` runs
                - **index** (list) :
                  [`n_bootstrap` x `S`] list containing trait indeces used in
                  each bootstrap run; to be passed to gradient computation

        """

        res = 0
        index = {}
        for i in range(n_bootstrap):
            index[i] = np.ix_(index_bootstrap[i], index_bootstrap[i])
            res += bn.ss(C_value[index[i]] - list_C[i], axis=None)
        return res, index

    @staticmethod
    def __RSS_grad_compute(n_bootstrap, n_params, C, C_value, list_C, index):
        r"""
        Compute gradient of residual sum of squares (rss) for each bootstrap
        run at each index

        - used as gradient parameter in quasi-Newton method (BFGS) of
          optimization
        - matrix to be optimized initialised with average over all bootstraps
          for each position

        Arguments:
            n_bootstrap (int):
                number of successful bootstrap runs;
            n_params (int):
                number of parameters of the model (parameters needed to build
                positive, semi-definite matrix with Choleski decomposition)
            C_value (array-like):
                [`P` x `P`] matrix to be optimized
            list_C (array-like):
                [`n_bootstrap` x `S` x `S`]  bootstrapped covariance matrices
            index (list):
                [`n_bootstrap `x `S`] trait indices used in each bootstrap run

        Returns:
            (numpy.array):

                - **res**:
                  [`n_params` x 1] sum of gradient over all successful
                  bootstrap runs for each parameter to be fitted.

        """
        res = sp.zeros(n_params)
        for pi in range(n_params):
            Cgrad = C.K_grad_i(pi)
            for i in range(n_bootstrap):
                res[pi] += (2 * Cgrad[index[i]] *
                            (C_value[index[i]] - list_C[i])).sum()
        return res

    def __VarianceDecomposition(self, phenoSubset, bs=None):
        r"""
        Compute variance decomposition of phenotypes into genetic and noise
        covariance

        Arguments:
            phenoSubset (array-like):
                [`N` x `S`] phenotypes for which variance decomposition should
                be computed
            bs (int):
                number of subsample
            phenotypes (array-like):
                [`N` x `P`] original phenotypes
            relatedness (array-like):
                [`N` x `N`] kinship/genetic relatedness used in estimation of
                genetic component
            output (string):
                output directory; needed for caching

        Returns:
            (dictionary):
                dictionary containing:

                - **Cg** (numpy.array):
                  [`S` x `S`] genetic variance component
                - **Cn** (numpy.array):
                  [`S` x `S`] noise variance component
                - **process_time** (double):
                  cpu time of variance decomposition
                - **bsindex** (int):
                  number of subsample
        """

        # time variance decomposition
        t0 = time.clock()
        vd = limix.mtset.MTSet(Y=phenoSubset, R=self.relatedness)
        vd_null_info = vd.fitNull(
            cache=False,
            n_times=self.iterations,
            rewrite=True)
        t1 = time.clock()
        processtime = t1 - t0

        if vd_null_info['conv']:
            verboseprint(
                ('Variance decomposition for bootstrap number {} '
                 'converged').format(bs + 1), verbose=self.verbose)
            Cg = vd_null_info['Cg']
            Cn = vd_null_info['Cn']
        else:
            verboseprint(
                ('Variance decomposition for bootstrap number {} '
                 'did not converge').format(bs + 1), verbose=self.verbose)
            Cg = None
            Cn = None
        return {'Cg': Cg, 'Cn': Cn, 'process_time': processtime, 'bsindex': bs}
