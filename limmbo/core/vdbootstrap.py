######################
### import modules ###
######################

import sys

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

import limix_legacy as dlimix
from limix.mtset import MTSet as MTST

import pp

######################
### core functions ###
#####################


class DataLimmbo(object):
    def __init__(self, datainput, verbose=False ):
        self.phenotypes = datainput.phenotypes
        self.relatedness = datainput.relatedness

    def generateBootstrapMatrix(self, P,
                                seed=12321,
                                p=10,
                                minCooccurrence=3,
                                n=None,
                                verbose=True):
        r"""
        Generate subsampling matrix.

        Arguments:
            seed (int,optional): 
                for pseudo-random numbers generation; default: 12321
            P (int): 
                total number of traits
            p (int): 
                size of bootstrap subset
            minCooccurrence (int): 
                minimum number a trait pair should be sampled; once reached 
                for all trait pairs, sampling is stopped if n is None; 
                default=3
            n (int): 
                if not None, sets the total number of permutations,
                otherwise n determined by minCooccurrence;  default: None
            verbose (bool):

        Returns:
            (tuple): 
                tuple containing:
 
             - **runs** (int): n if n was not None, or determined once all 
               trait-trait subsamplings have occurrd minCooccurence times
             - **return_list** (list): of length self.runs containing [1 x S] 
               (np.array) with sample of numbers range(P)
             - **counts_min** (int): minimum trait-trait co-occurrence in 
               sampling matrix
        """
        rand_state = np.random.RandomState(seed)
        counts = sp.zeros((P, P))
        return_list = []

        if n is not None:
            verboseprint(
                'Generate bootstrap matrix with {} bootstrap samples',
                 '(number of specified bootstraps'.format(n),
                verbose=verbose)
            for i in xrange(n):
                bootstrap = rand_state.choice(
                    a=range(P),
                    size=p,
                    replace=False)
                return_list.append(bootstrap)
                index = np.ix_(np.array(bootstrap), np.array(bootstrap))
                counts[index] += 1
            self.runs = n
        else:
            while counts.min() < minCooccurrence:
                bootstrap = rand_state.choice(
                    a=range(P),
                    size=p,
                    replace=False)
                return_list.append(bootstrap)
                index = np.ix_(np.array(bootstrap), np.array(bootstrap))
                counts[index] += 1
            self.runs = len(return_list)
            verboseprint(
                'Generated bootstrap matrix with {} bootstrap runs '
                 ' such that each trait-trait was sampled {}'.format(self.runs,    
                     minCooccurrence),
                verbose=verbose)
        
        bootstrap_matrix = np.array(return_list)
        return bootstrap_matrix, self.runs, counts.min()

    def bootstrapPhenotypes(self, bs, bootstrap_matrix):
        r"""
        Subsample [S] phenotypes with [N] samples form total of [P]
        phenotypes. Indeces for subsampling provided in [bs x S]
        bootstrap_matrix, where bs is the bs is the total number of bootstraps
        as determined by .generateBootstrapMatrix()

        Arguments:
            bs (int): 
                bootstrap index
            bootstrap_matrix (pandas.DataFrame):
                [bs x S] with subsampling indeces for phenotypes

        Returns:
            (numpy.array):

                - **phenotypes**: [N x S] of subsampled phenotypes

        Examples:

             .. doctest::

             >>> import pandas
             >>> import numpy
             >>> from numpy.random import RandomState
             >>> from limmbo.io.input import InputData
             >>> from limmbo.core.vdbootstrap import DataLimmbo
             >>> from numpy.linalg import cholesky as chol
             >>> random = RandomState(10)
             >>> P = 50
             >>> p = 10
             >>> N = 100
             >>> S = 1000
             >>> snps = (random.rand(N, S) < 0.2).astype(float)
             >>> kinship = numpy.dot(snps, snps.T)/float(10)
             >>> y  = random.randn(N,P) 
             >>> pheno = numpy.dot(chol(kinship),y)
             >>> pheno_ID = [ 'PID{}'.format(x+1) for x in range(P)]
             >>> samples = [ 'SID{}'.format(x+1) for x in range(N)]
             >>> datainput = InputData()      
             >>> datainput.addPhenotypes(phenotypes = pheno,        
             ...                         phenotype_ID = pheno_ID,             
             ...                         pheno_samples = samples)          
             >>> datainput.addRelatedness(relatedness = kinship,        
             ...                          relatedness_samples = samples)      
	     >>> limmbo = DataLimmbo(datainput = datainput)
	     >>> bootstrap_matrix, r, c= limmbo.generateBootstrapMatrix(P,
             ...                                                verbose=False)
	     >>> bootstrap_pheno_1 = limmbo.bootstrapPhenotypes(bs=0, 
	     ...                           bootstrap_matrix = bootstrap_matrix)
        """
        bootstrap = bootstrap_matrix[bs, :]
        phenotypes = self.phenotypes[:, bootstrap]

        return phenotypes

    def sampleCovarianceMatricesPP(self, runs, P, p, output, seed, cpus,
            verbose=False):
        r"""
        Distribute variance decomposition of subset matrices via pp
        
        Arguments:
            runs (int):
                number of bootstrapping runs executed for this experiment
            P (int): 
                number of phenotypes P
            p (int):  
                subsampling size S, default: 10
            seed (int): 
                seed to initialise random number generator for bootstrapping
            minCooccurrence (int):
                minimum number a trait pair should be sampled; once reached 
                for all trait pairs, sampling is stopped if n is None; 
                default=3
            n (int): 
                if not None, sets the total number of permutations,
                otherwise n determined by minCooccurrence;  default: None
            output (string): 
                output directory; needs writing permission
            cpus (int): 
                number of cpus available for covariance estimation
            verbose (bool): 
                should messages be printed to stdout
        
        Returns:
            (tuple): 
                tuple containing:
 
                - **results** (list): containing results of 
                  .VarianceDecomposition of the subset matrices
                - **bootstrap_matrix** (pandas.DataFrame): sdfs
        """

        runs, bootstrap_matrix, minimumTraitTraitcount = \
		self.generateBootstrapMatrix(seed=seed,
                        n=runs,
                        P=P, p=p,
                        minCooccurrence=minCooccurrence)
        bootstrap_matrix = pd.DataFrame(bootstrap_matrix)
        bootstrap_matrix.to_csv(
            '{}/bootstrap_matrix.csv'.format(output),
            sep=",",
            index=True,
            header=False)
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
            verbose=verbose)

        nrtraits, nrsamples = phenotypes.shape
        outfile = "%s/mtSetresults_nrsamples%s_nrtraits%s.h5" % (
            output, nrsamples, nrtraits)
        for bs in range(self.runs):
            pheno = self.bootstrapPhenotypes(bs, bootstrap_matrix)
            verboseprint('Start vd for bootstrap nr {}'.format(bs))
            jobs.append(
                job_server.submit(self.VarianceDecomposition, (pheno, bs),
                                  (verboseprint, ), ("mtSet", "time")))

        for job in jobs:
            bsresult = job()
            bsresult['bootstrap'] = bootstrap_matrix.iloc[bsresult[
                'bsindex'], :]
            results.append(bsresult)

        return results, bootstap_matrix

    def combineBootstrap(self, results, timing=True):
        r"""
        Combine the [S x S] subset covariance matrices to find the overall
        [P x P] covariance matrices Cg and Cn and write as .csv files

        Arguments:
            results (list): 
		results of sampleCovarianceMatricesPP()
            timing (bool):
		 should runtime be recorded and written to file
            output (string): 
		output directory; needs writing permission
            seed (int): 
		seed to initialise random number generator for bootstrapping

        Returns:
            self
        """
        verboseprint('Combine bootstrapping results...', verbose=verbose)
        time0 = time.clock()
        Cg_fit, Cn_fit, Cg_average, Cn_average, process_time_bs, nr_bs = \
                self.getBootstrapResults(results=results,
                        timing=timing)
        time1 = time.clock()

        if timing is True:
            proc_time_combine_bs = time1 - time0
            proc_time_sum_ind_bs = np.array(process_time_bs).sum()
            pd.DataFrame(process_time_bs).to_csv(
                "%s/process_time_bs.csv" % (self.options.output),
                sep=",",
                header=False,
                index=False)
            pd.DataFrame(
                [proc_time_combine_bs, proc_time_sum_ind_bs],
                index=["Proctime combine BS",
                       "Proctime sum of individual BS"]).to_csv(
                           "%s/process_time_summary.csv" %
                           (self.options.output),
                           sep=",",
                           header=False,
                           index=True)

        Cg_average, Cg_average_ev_min = regularize(Cg_average)
        Cn_average, Cn_average_ev_min = regularize(Cn_average)
        Cg_fit, Cg_fit_ev_min = regularize(Cg_fit)
        Cn_fit, Cn_fit_ev_min = regularize(Cn_fit)

        verboseprint("Generate output files", verbose=self.options.verbose)
        pd.DataFrame(Cg_average).to_csv(
            "%s/Cg_average_seed%s.csv" % (self.options.output,
                                          self.options.seed),
            sep=",",
            header=False,
            index=False)
        pd.DataFrame(Cn_average).to_csv(
            "%s/Cn_average_seed%s.csv" % (self.options.output,
                                          self.options.seed),
            sep=",",
            header=False,
            index=False)

        pd.DataFrame(Cg_fit).to_csv(
            "%s/Cg_fit_seed%s.csv" % (self.options.output, self.options.seed),
            sep=",",
            header=False,
            index=False)
        pd.DataFrame(Cn_fit).to_csv(
            "%s/Cn_fit_seed%s.csv" % (self.options.output, self.options.seed),
            sep=",",
            header=False,
            index=False)

        return self

    def VarianceDecomposition(self, phenoSubset, cache=False, output=None, 
            bs=None, iterations=10, verbose=False):
        r"""
	Compute variance decomposition of phenotypes into genetic and noise
        covariance

        Arguments:
            phenoSubset (array-like): 
                [N x S] phenotypes for which variance decomposition should be
                computed
            bs (int): 
                number of subsample
            phenotypes (array-like): 
                [N x P] original phenotypes
            iterations (int):
                
            relatedness (array-like): 
                [N x N] kinship/genetic relatedness used in estimation of 
                genetic component
            output (string): 
                output directory; needed for caching
            cache (bool): 
                should variance decomposition results be cached to output
            verbose (bool): 
                should messages be printed to stdout
        
        Returns:
            (dictionary): 
                dictionary containing:
 
                - **Cg** (numpy.array): [S x S] genetic variance component
                - **Cn** (numpy.array): [S x S] noise variance component
                - **process_time** (double): cpu time of variance decomposition
                - **bsindex** (int): number of subsample
        """

        outfile = None
        if cache is True and output is None:
            sys.exit(("Output directory must be specified if caching is "
                      "enabled"))
        if cache is False and output is not None:
            print("Warning: Caching is disabled, despite having supplied an "
                  "output directory")
        if cache is True and output is not None:
            nrtraits, nrsamples = phenotypes.shape
            outfile = "%s/mtSetresults_nrsamples%s_nrtraits%s.h5" % (
                output, nrsamples, nrtraits)

        # time variance decomposition
        t0 = time.clock()
        vd = MTST(Y=phenoSubset, R=relatedness)
        vd_null_info = vd.fitNull(
            cache=cache,
            fname=outfile,
            n_times=iterations,
            rewrite=True)
        t1 = time.clock()
        processtime = t1 - t0

        if vd_null_info['conv']:
            verboseprint(
                'Variance decomposition for bootstrap number {}', 
                'converged'.format(bs), verbose=verbose)
            Cg = vd_null_info['Cg']
            Cn = vd_null_info['Cn']
        else:
            verboseprint(
                'Variance decomposition for bootstrap number {}', 
                'did not converge'.format(bs), verbose=verbose)
            Cg = None
            Cn = None
        return {'Cg': Cg, 'Cn': Cn, 'process_time': processtime, 'bsindex': bs}

    def getBootstrapResults(self, results, runs, P, p, output=None, 
            verbose=False, pickle=False):
        r"""
        Collect bootstrap results of [S x S] traits and combine all runs to
        total [P x P] covariance matrix
        
        Arguments:
            results (list): 
                results of sampleCovarianceMatricesPP()
            runs (int):
                number of bootstrapping runs executed for this experiment
            P (int): 
                number of phenotypes P
            p (int):  
                subsampling size S, default: 10
            output (string): 
                output directory; needed for pickling all variance 
                decomposition runs of [S x S] Cg and Cn
            verbose (bool): 
                should messages be printed to stdout

        Returns:    
            (tuple): 
                tuple containing:
 
                - **Cg_opt** (numpy.array): [P x P] genetic covariance matrix 
                  via fitting
                - **Cn_opt** (numpy.array): [P x P] noise covariance matrix via
                  fitting
                - **Cg_norm** (numpy.array): [P x P] genetic covariance matrix 
                  via simple average
                - **Cn_norm** (numpy.array): [P x P] noise covariance matrix 
                  via simple average
                - **process_time_bs** (list): run times for all variance
                  decomposition runs of [S x S] Cg and Cn
                - **number_of_bs** (int): total number of successful 
                  bootstrapping runs
        """

        # list to contain trait indeces of each bootstrap run
        bootstrap = {}
        #sample_ID = None

        # create np.arrays of dimension PxP for mean and std of each entry in
        # covariance matrix
        Cg_norm = sp.zeros((P, P))
        Cn_norm = sp.zeros((P, P))

        # Process time per bootstrap
        process_time_bs = []

        # create np.arrays of dimension runs x P x P to store the bootstrapping
        # results for averaging
        Cg_average = nans((runs, P, P))
        Cn_average = nans((runs, P, P))

        Cg_fit = nans((runs, p, p))
        Cn_fit = nans((runs, p, p))

        n = 0
        for vdresult in results:
            bootstrap[n] = vdresult['bootstrap'].values
            if self.options.timing == True:
                process_time_bs.append(vdresult['process_time'])

            # store results of each bootstrap as matrix of inflated
            # matrices: NAs for traits that were not sampled
            if vdresult['Cg'] is None or vdresult['Cn'] is None:
                continue
            Cg_average[n, :, :] = inflate_matrix(
                vdresult['Cg'], bootstrap[n], P=P, zeros=False)
            Cn_average[n, :, :] = inflate_matrix(
                vdresult['Cn'], bootstrap[n], P=P, zeros=False)

            # store results of each bootstrap as matrix of small p x p
            # matrices generated by each bootstrap
            Cg_fit[n, :, :] = vdresult['Cg']
            Cn_fit[n, :, :] = vdresult['Cn']
            n += 1
        # Total number of successful bootstrapping runs
        number_of_bs = n - 1

        if pickle:
            cPickle.dump(Cg_fit,
                     open('{}/Cg_all_bootstraps.p'.format(output), "wb"))
            cPickle.dump(Cn_fit,
                     open('{}/Cn_all_bootstraps.p'.format(output), "wb"))
        # Computing  mean and standard deviation of bootstrapping results at
        # each position p1, p2 in overall PxP covariance matrix
        verboseprint(
            ("Computing mean of bootstrapping results"),
            verbose=verbose)
        for p1 in range(P):
            for p2 in range(P):
                vals = Cg_average[:, p1, p2]
                Cg_norm[p1, p2] = vals[~sp.isnan(vals)].mean()
                vals = Cn_average[:, p1, p2]
                Cn_norm[p1, p2] = vals[~sp.isnan(vals)].mean()

        verboseprint(
            ("Fitting bootstrapping results: minimize residual sum of"
             "squares over all bootstraps"),
            verbose=verbose)
        Cg_opt = self.fit_bootstrap_results(
            cov_init=Cg_norm,
            cov_bootstrap=Cg_fit,
            bootstrap_indeces=bootstrap,
            number_of_bs=number_of_bs,
            name="Cg")
        Cn_opt = self.fit_bootstrap_results(
            cov_init=Cn_norm,
            cov_bootstrap=Cn_fit,
            bootstrap_indeces=bootstrap,
            number_of_bs=number_of_bs,
            name="Cn")

        return Cg_opt, Cn_opt, Cg_norm, Cn_norm, process_time_bs, number_of_bs

    def fit_bootstrap_results(self, cov_init, cov_bootstrap, bootstrap_indeces,
                              number_of_bs, name):
        r"""
        Fit  bootstrap results of [S x S] traits and combine all runs to total
        [P x P] covariance matrix.

        Arguments:
            lp (int): 
		number of phenotypes P
            cov_init (array-like): 
		[P x P] covariance matrix used to initialise fitting
            number_of_bs (int): 
		number of successful bootstrapping runs executed for this
		experiment
            cov_bootstrap (array-like): 
		[number_of_bs x S x S] bootstrap results
            bootstrap_indeces (array-like): 
		[number_of_bs x S] matrix of  bootstrapping indeces
            name (string): name of covariance matrix

        Returns:
            (array-like): 

                - **C_opt_value** (array-like): 
                  [P x P] covariance matrix if fit successful, else 1x1 matrix
                  containing string 'did not converge'


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
        verboseprint(
            "Fitting parameters (minimizing rss via BFGS)...",
            verbose=self.options.verbose)
        res = opt(
            self.rss,
            x0=params,
            args=(C_fit, number_of_bs, bootstrap_indeces, cov_bootstrap),
            factr=1e12,
            iprint=2)
        cPickle.dump(res,
                     open("%s/%s_opt_result.p" % (self.options.output, name),
                          "wb"))

        if res[2]['warnflag'] == 0:
            C_opt = dlimix.CFreeFormCF(self.options.P)
            C_opt.setParams(res[0])
            C_opt_value = C_opt.K()
        else:
            C_opt_value = np.array(['did not converge'])
        return C_opt_value

    def rss(self, params, C, n_bootstrap, index_bootstrap, list_C):
        r"""
        Compute residual sum of squares and gradient for estimate of covariance
        matrix C and bootstrapped values of C.

        Arguments:
            params (array-like):
		of parameters used for initialising estimate C 
		(length: 1/2*P*(P+1))
            C: 
		intialised free-form covariance matrix, to be fitted
            n_bootstrap (int): 
		number of successful bootstrapping runs executed for this
		experiment
            index_bootstrap (numpy.array): 
		[number_of_bs x S] matrix of bootstrapping indeces
            list_C (list):
		list of n_bootstrap [p x p] bootstrapped covariance matrices

        Returns:
            (tuple): 
                tuple containing:

                - **RSS_res**: residual sum of squares
                - **RSS_grad_res**: gradient of residual sum of squares 
                  function

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
        RSS_res, index = self.RSS_compute(n_bootstrap, C_value, list_C,
                                          index_bootstrap)
        # compute gradient (first derivative of rss at each bootstrap)
        RSS_grad_res = self.RSS_grad_compute(n_bootstrap, n_params, C, C_value,
                                             list_C, index)

        return RSS_res, RSS_grad_res

    @staticmethod
    def RSS_compute(n_bootstrap, C_value, list_C, index_bootstrap):
        r"""
        Compute residual sum of squares (rss) for each bootstrap run
        -used parameter to be optimized  in quasi-Newton method (BGS) of
        optimzation
        -bootstrap matrices versus matrix to be optimized
        -matrix to be optimized initialised with average over all bootstraps
        for each position
        
        Arguments:
            n_bootstrap (int): 
                number of successful bootstrap runs;
            C_value (array-like): 
                [P x P] matrix to be optimized
            list_C (array-like): 
                [n_bootstrap x p x p]  bootstrapped covariance matrices
        
        Returns:
            (tuple): 
                tuple containing:
 
                - **res** (double) : added residual sum of squares over all
                  n_bootstrap runs
                - **index** (list) : [n_bootstrap x p] list containing trait 
                  indeces used in each bootstrap run; to be passed to 
                  gradient computation

        """
        
        res = 0
        index = {}
        for i in range(n_bootstrap):
            index[i] = np.ix_(index_bootstrap[i], index_bootstrap[i])
            res += bn.ss(C_value[index[i]] - list_C[i], axis=None)
        return res, index

    @staticmethod
    def RSS_grad_compute(n_bootstrap, n_params, C, C_value, list_C, index):
        r"""
        Compute gradient of residual sum of squares (rss) for each bootstrap
        run at each index

        - used as gradient parameter in quasi-Newton method (BFGS) of
          optimzation
        - matrix to be optimized initialised with average over all bootstraps
          for each position
        
        Arguments:
            n_bootstrap (int): 
                number of successful bootstrap runs;
            n_params (int): 
                number of parameters of the model (parameters needed to build
                positive, semi-definite matrix with Choleski decomposition)
            C_value (array-like): 
                [P x P] matrix to be optimized
            list_C (array-like): 
                [n_bootstrap x p x p]  boostrapped covariance matrices 
            index (list): 
                [n_bootstrap x p] trait indeces used in each bootstrap run

        Returns:
            (numpy.array):
                - **res**: [n_params x 1] sum of gradient over all bootstrap 
                  runs for each parameter to be fitted 

        """
        res = sp.zeros(n_params)
        for pi in range(n_params):
            Cgrad = C.Kgrad_param(pi)
            for i in range(n_bootstrap):
                res[pi] += (2 * Cgrad[index[i]] *
                            (C_value[index[i]] - list_C[i])).sum()
        return res
