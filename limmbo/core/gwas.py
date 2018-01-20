import sys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import copy

import scipy as sp
import pandas as pd
import numpy as np

import limix as limi
import limix.qtl as qtl
import limix.plot as plot
import limix.mtset

from limmbo.utils.utils import verboseprint
from limmbo.utils.utils import effectiveTests


class GWAS(object):
    """
    """
    def __init__(self, datainput, seed=10, verbose=True,
            searchDelta=False):
        '''
        nothing to initialize
        '''
        self.verbose = verbose
        self.seed = seed
        #self.meff = meff
        self.searchDelta = searchDelta
        self.genotypes = datainput.genotypes
        self.genotypes_info = datainput.genotypes_info
        self.phenotypes = datainput.phenotypes
        self.phenotype_ID = datainput.phenotype_ID
        self.covariates = datainput.covariates
        self.relatedness = datainput.relatedness
        self.pcs = datainput.pcs
        self.Cg = datainput.Cg
        self.Cn = datainput.Cn
        self.test = "lrt"

        self.pvalues = None
        self.pvalues_adjust = None
        self.pvalues_empirical_raw = None
        self.stats = None
        self.z = None
        model = None
        self.pvalues_empirical = None
        self.adjustBy = None


    ############################
    ### core functions GWAS: ###
    ############################

    def runAssociationAnalysis(self, mode, setup="lmm", adjustSingleTrait=None):
        r"""
        Analysing the association between phenotypes, genotypes, optional
        covariates and random genetic effects.

        Arguments:
            mode (string):
                specifies the type of linear model: either 'multitrait' for
                multivariate analysis or 'singletrait' for univariate analysis.
            setup (string):
                specifies the linear model setup: either 'lmm' for linear mixed
                model or 'lm' for a simple linear model.
            adjustSingleTrait (string):
                Method to adjust single-trait association p-values for testing
                multiple traits; If None (default) no adjusting. Options are
                'bonferroni' (for bonferroni correction') or 'effective' (for
                correcting for the effective number of tests as described in 
                `(Galwey,2009)
                <http://onlinelibrary.wiley.com/doi/10.1002/gepi.20408/abstract>`_.
                

        Returns:
            (dictionary):
                dictionary containing:

                - **lm** (:class:`limix.qtl.LMM`):
                  LIMIX LMM object
                - **pvalues** (numpy array):
                  [`NrSNP` x `P`] (when mode is singletrait) or [1 x`NrSNP`]
                  array of p-values.
                - **betas** (numpy array):
                  [`NrSNP` x `P`] array of effect size estimates per SNP across
                  all traits.
                - **pvalues_adjust** (numpy array):
                  only returned if mode is 'singletrait' and 'adjustSingleTrait'
                  is not None; contains single-trait p-values adjusted for the
                  number of single-trait analyses conducted.

        Examples:

            .. doctest::

                >>> from pkg_resources import resource_filename
                >>> from limmbo.io.reader import ReadData
                >>> from limmbo.io.input import InputData
                >>> from limmbo.core.gwas import GWAS
                >>> data = ReadData()
                >>> file_pheno = resource_filename('limmbo',
                ...                                'io/test/data/pheno.csv')
                >>> file_geno = resource_filename('limmbo',
                ...                                'io/test/data/genotypes.csv')
                >>> file_relatedness = resource_filename('limmbo',
                ...                     'io/test/data/relatedness.csv')
                >>> file_covs = resource_filename('limmbo',
                ...                               'io/test/data/covs.csv')
                >>> file_Cg = resource_filename('limmbo',
                ...                     'io/test/data/Cg.csv')
                >>> file_Cn = resource_filename('limmbo',
                ...                     'io/test/data/Cn.csv')
                >>> data.getPhenotypes(file_pheno=file_pheno,
                ...                   verbose=False)
                >>> data.getCovariates(file_covariates=file_covs,
                ...                   verbose=False)
                >>> data.getRelatedness(file_relatedness=file_relatedness,
                ...                     verbose=False)
                >>> data.getGenotypes(file_geno=file_geno,
                ...                   verbose=False)
                >>> data.getVarianceComponents(file_Cg=file_Cg,
                ...                            file_Cn=file_Cn,
                ...                            verbose=False)
                >>> indata = InputData()
                >>> indata.addPhenotypes(phenotypes = data.phenotypes,
                ...                      pheno_samples = data.pheno_samples,
                ...                      phenotype_ID = data.phenotype_ID)
                >>> indata.addRelatedness(relatedness = data.relatedness,
                ...     relatedness_samples = data.relatedness_samples)
                >>> indata.addCovariates(covariates = data.covariates,
                ...                      covs_samples = data.covs_samples)
                >>> indata.addGenotypes(genotypes=data.genotypes,
                ...                     genotypes_info=data.genotypes_info,
                ...                     geno_samples=data.geno_samples)
                >>> indata.addVarianceComponents(Cg = data.Cg, Cn=data.Cn)
                >>> indata.commonSamples()
                >>> indata.regress(regress=True, verbose=False)
                >>> gwas = GWAS(datainput=indata, seed=10, verbose=False)
                >>>
                >>>
                >>> # Example of multi-trait single-variant association testing
                >>> # using a linear mixed model.
                >>> resultsAssociation = gwas.runAssociationAnalysis(
                ...     setup="lmm", mode="multitrait")
                >>> resultsAssociation.keys()
                ['lm', 'betas', 'pvalues']
                >>> resultsAssociation['pvalues'].shape
                (1, 20)
                >>> '{:0.3e}'.format(resultsAssociation['pvalues'].min())
                '9.055e-09'
                >>> resultsAssociation['betas'].shape
                (10, 20)
                >>>
                >>>
                >>> # Example of single-trait single-variant association testing
                >>> # using a linear mixed model.
                >>> resultsAssociation = gwas.runAssociationAnalysis(
                ...     setup="lmm", mode="singletrait",
                ...     adjustSingleTrait = "effective")
                >>> resultsAssociation.keys()
                ['pvalues_adjust', 'lm', 'betas', 'pvalues']
                >>> resultsAssociation['pvalues'].shape
                (10, 20)
                >>> resultsAssociation['betas'].shape
                (10, 20)
                >>> '{:0.3e}'.format(resultsAssociation['pvalues_adjust'].min())
                '1.037e-02'
        """

        # set parameters for the analysis
        self.N, self.P = self.phenotypes.shape
        self.S    = self.genotypes.shape[1]
        verboseprint("Loaded {} samples, {} phenotypes, {} snps".format(
            self.N, self.P, self.S), verbose = self.verbose)
        verboseprint("Set searchDelta {}".format(self.searchDelta),
            verbose = self.verbose)
            
        self.setup = setup
        self.mode = mode

        if mode == "multitrait":
            associationResults = self.__multiTraitAssociation_anyeffect(
                genotypes = self.genotypes)

        if mode == "singletrait":
            associationResults = self.__singleTraitAssociation(
                genotypes = self.genotypes,
                adjustSingleTrait = adjustSingleTrait)

        return associationResults

    def __multiTraitAssociation_anyeffect(self, genotypes, empiricalP=False,
            computeFDR=False):
        r"""
        Muti-trait association test, testing for an effect of the genotype on
        `any` phenotype (`any` effect test). Wraps around
        `qtl.qtl_test_lmm_kronecker`.

        Arguments:
            genotypes (array-like):
                [`N` x `NrSNP`] genotype array of [`N`] samples and [`NrSNP`]
                genotypes
            empiricalP (bool):
                set to True if association test is part of estimating empirical
                pvalues
            computeFDR (bool):
                set to True if association test is part of estimating empirical
                FDR

        Returns:
            (dictionary):
                dictionary containing:

                - **lm**(:class:`limix.qtl.LMM`):
                  LIMIX LMM object
                - **pvalues** (numpy array):
                  [1 x`NrSNP`] array of
                  p-values
                - **betas** (numpy array):
                  [`NrSNP` x `P`] array of effect size estimates per SNP across
                  all traits

        """
        if self.covariates is None:
            Acovs = None
        else:
            Acovs = sp.eye(self.P)
        if self.setup is "lmm":
            if self.Cg is None:
                if P > 30:
                    print("Warning: For large trait sizes, computation times "
                  "for pure REML variance decomposition are long, "
                   "consider bootstrapping trait-trait covariance"
                    " components")
                verboseprint("Estimate Variance components",
                            verbose = self.verbose)
                self.Cg, self.Cn = self.__varianceDecomposition()
            K1c = self.Cg
            K2c = self.Cn
            K1r = self.relatedness
            self.model="lmm_mt"
        else:
            K1c = 1e-9*sp.eye(self.P)
            K2c = sp.cov(self.phenotypes.T)
            K1r = sp.eye(self.N)

            if self.pcs is not None:
                self.model="lm_mt_pcs"
            else:
                self.model="lm_mt"
        Asnps = sp.eye(self.P)
        if not empiricalP and not computeFDR:
            verboseprint("Computing multi-trait (any effect) model: {}".format(
                self.model), verbose = self.verbose)

        lm, pvalues = qtl.qtl_test_lmm_kronecker(snps=genotypes,
                phenos=self.phenotypes, Asnps=Asnps, Acovs=Acovs,
                covs=self.covariates, K1r=K1r, K1c=K1c, K2c=K2c,
                searchDelta=self.searchDelta)

        if not empiricalP and not computeFDR:
            betas = lm.getBetaSNP()
        else:
            betas = None
            lm = None
        
        return {"pvalues": pvalues, "betas": betas, "lm": lm}


    def __singleTraitAssociation(self, genotypes, adjustSingleTrait=None,
            empiricalP=False, computeFDR=False):
        r"""
        Single-trait association test. Wraps around `qtl.qtl_test_lmm`.

        Arguments:
            genotypes (array-like):
                [`N` x `NrSNP`] genotype array of [`N`] samples and [`NrSNP`]
                genotypes
            empiricalP (bool):
                set to True if association test is part of estimating empirical
                pvalues
            computeFDR (bool):
                set to True if association test is part of estimating empirical
                FDR
            adjustSingleTrait (string):
                Method to adjust single-trait association p-values for testing
                multiple traits; If None (default) no adjusting. Options are
                'bonferroni' (for bonferroni correction') or 'effective' (for
                correcting for the effective number of tests as described in 
                `(Galwey,2009)
                <http://onlinelibrary.wiley.com/doi/10.1002/gepi.20408/abstract>`_

        Returns:
            (dictionary):
                dictionary containing:

                    - **lm**(:class:`limix.qtl.LMM`):
                      LIMIX LMM object
                    - **pvalues** (numpy array):
                      [`P` x `NrSNP`] array of p-values
                    - **betas** (numpy array):
                      [`P` x `NrSNP`] array of effect size estimates per SNP 
                      across all traits
                    - **pvalues_adjust** (numpy array):
                      only returned if 'adjustSingleTrait' is not None; contains
                      single-trait p-values adjusted for the number of 
                      single-trait analyses conducted

        """

        if self.setup is "lmm":
            self.model="lmm_st"
            K = self.relatedness
        else:
            if self.pcs is not None:
                self.model="lm_st_pcs"
            else:
                self.model="lm_st"
                K = None

        if not empiricalP and not computeFDR:
            verboseprint("Computing single-trait association ({})".format(
                self.model), verbose = self.verbose)
            lm = qtl.qtl_test_lmm(snps=genotypes, pheno=self.phenotypes,
                K=K, covs=self.covariates, test=self.test)
            pvalues = lm.getPv()

        if not empiricalP and not computeFDR:
            betas = lm.getBetaSNP()
        else:
            betas = None
            lm = None

        if adjustSingleTrait is not None:
            if self.adjustBy is None:
                if adjustSingleTrait is "bonferroni":
                    self.adjustBy = self.P
                elif adjustSingleTrait is "effective":
                    self.adjustBy = effectiveTests(self.phenotypes)
                else:
                    raise (("{} is not a provided method to adjust single-trait"
                    "pvalues for multiple hypothesis testing").format(
                        adjustSingleTrait))
            pvalues_adjust = np.array([self.__adjust(p) for p in pvalues])
        else:
            pvalues_adjust = None

        self.adjustSingleTrait = adjustSingleTrait

        return {"lm":lm, "pvalues": pvalues, "pvalues_adjust": pvalues_adjust,
                "betas": betas}


    def saveAssociationResults(self, model, output, chromosome, columns=None,
            plotResults=False):
        r"""

        """

        outstring = (output, model, chromosome)
        
        self.genotypes_info['SNP'] = self.genotype_info.index

        beta_df = pd.DataFrame(results['betas'].T,
            index=gwas.genotypes_info.index, columns = columns)
        beta_df = pd.concat([self.genotypes_info, beta_df], axis=1)
        
        if self.mode is 'singletrait':
            pvalues_df = pd.DataFrame(results['pvalues'].T,  
                index=gwas.genotypes_info.index, columns = columns)

            if results['pvalues_adjust'] is not None:
                pvalues_adjust_df = pd.DataFrame(results['pvalues_adjust'].T,  
                    index=gwas.genotypes_info.index, columns = columns)
                
                pvalues_adjust_df.to_csv("%s/%s_padjust_%s.csv" % outstring,
                    index=False)
            
            if results['pvalues_empirical'] is not None:
                
                pempirical_df = pd.DataFrame(results['pvalues_empirical'].T,
                    index=gwas.genotypes_info.index, columns = columns)
                    #index=SNP, columns=['Pempirical'])
                
                pempirical_raw_df = pd.DataFrame(self.pvalues_empirical_raw.T,
                    index=gwas.genotypes_info.index, columns = columns)
                pempirical_raw_df.to_csv("%s/%s_pempirical_raw%s%s.csv" %
                    (outstring + (self.fdr,)), index=False)
            
        if self.mode is 'multitrait':
            pvalues_df = pd.DataFrame(results['pvalues'],
                index=gwas.genotypes_info.index, columns = "P")
            
            if self.pvalues_empirical is not None:
                pvalue_df = pd.DataFrame(results['pvalues_empirical'],
                    index=gwas.genotypes_info.index, columns = "P")

            
        pvalues_df.to_csv("%s/%s_pvalue_%s.csv" % outstring, index=False)
        beta_df.to_csv("%s/%s_betavalue_%s.csv" % outstring, index=False)

        if results['pvalues_empirical'] is not None:
            pempirical_df.to_csv("%s/%s_pempirical_%s%s.csv" %
                (outstring + (self.fdr,)), index=False)


        if plotResults:
            self.manhattanQQ(model=model, P=P)
        
        if self.estimate_vd:
            if self.timeVD is not None:
                pd.DataFrame(self.timeVD).to_csv(
                    "%s/timeVarianceDecomposition_REML.csv" % output,
                    index=False, header=False)

            pd.DataFrame(self.Cg).to_csv("%s/Cg_REML.csv" % (output),
                index=False, header=False)
            pd.DataFrame(self.Cn).to_csv("%s/Cn_REML.csv" % (output),
                index=False, header=False)

            if self.fdr is not None:
                pd.DataFrame(self.ppermute).to_csv("%s/%s_ppermute_%s.csv" %
                    outstring, index=False)
                pd.DataFrame(['FDR', str(self.FDR)]).T.to_csv(
                    "%s/%s_empiricalFDR_%s.csv" % outstring, header=False, 
                    index=False)

    def computeEmpiricalP(self, pvalues, nrpermutations = 1000):
        r"""
        Compute empirical p-values: permute the genotypes, do the 
        association test, record if permuted p-value of SNP is smaller than
        original p-value. Sum these occurrences and divide by total number of
        permutation.

        Arguments:
            pvalues (array-like):
                [`P` x `NrSNP`] (single-trait) or [1 x `NrSNP`] (multi-trait) 
                array of p-values
            nrpermutations (int):
                number of permutations; 1/nrpermutations is the maximum level
                of significance (alpha)to test for, e.g. 
                nrpermuations=100 -> alpha=0.01 

        Returns:
            (numpy array): 
                [`P` x `NrSNP`] (single-trait) or [1 x `NrSNP`] (multi-trait) 
                array of emprirical p-values
        """

        verboseprint("Computing empirical p-values", verbose=self.verbose)
       
        np.random.seed(seed=self.seed)

        self.nrpermutations = nrpermutations
        pvalues = np.array(pvalues)

        if self.mode is "multitrait":
            count = np.zeros((nrpermutations, self.S))
        else:
            count = np.zeros((nrpermutations, self.P, self.S))

        for ps in range (nrpermutations):
            verboseprint("Permutation %s" % ps)
            genotypes_permute = self.genotypes[np.random.choice(self.N, self.N, 
                replace=False), :]
            if self.mode is "multitrait":
                resultsPermute = self.__multiTraitAssociation_anyeffect( 
                    genotypes = genotypes_permute, empiricalP=True)
                count[ps, :] = resultsPermute['pvalues'] <= pvalues
            if self.mode is "singletrait":
                resultsPermute = self.__singletraitAssociation(
                    genotypes = genotypes_permute, empiricalP=True)
                count[ps,:, :] = resultsPermute['pvalues'] <= pvalues


        pvalues_empirical = np.sum(count, axis=0)/nrpermutations
        
        return pvalues_empirical

    def computeFDR(self, fdr):
        r"""
        Create an empirical p-values distribution by permuting the genotypes
        and running the association test on these (10 random permuations). 
        Recored the observed pvalues, order by rank and find the rank of the 
        desired FDR to determine the empirical FDR.

        Arguments:
            fdr (float):
                desired fdr threshold

        Returns:
            (dictionary):
                dictionary containing:

                    - **fdr** (float):
                      empirical FDR
                    - **empirical_pvalue_distribution** (numpy array):
                      array of empirical p-values for 10 permutations tests
        """

        verboseprint("Computing empirical FDR", verbose = self.verbose)
        
        np.random.seed(seed=self.seed)
        #tests = int(1/self.options.fdr)
        tests = 10

        self.ppermute = np.zeros((tests, self.S))

        for ps in range (tests):
            genotypes_permute = self.genotypes[np.random.choice(self.N, self.N,
                replace=False), :]
            if self.mode == "multitrait":
                resultsFDR = self.__multiTraitAssociation_anyeffect(
                        genotypes = genotypes_permute, computeFDR=True)
                ppermute[ps, :] = resultsFDR['pvalues_permute']
            if self.mode == "singletrait":
                resultsFDR = self.__singleTraitAssociation(
                    genotypes = genotypes_permute, computeFDR=True,
                    adjustSingleTrait=adjustSingleTrait)
                pvalues_adjust_min = resultsFDR['pvalues_permute_adjust'].min(
                    axis=0)
                ppermute[ps, :] = pvalues_adjust_min

        SNPsPassingFDR = int(fdr* self.S* tests)
        self.allppermute = ppermute.flatten()
        self.allppermute.sort()
        self.fdr_empirical = self.allppermute[SNPsPassingFDR]

        return {"fdr": self.fdr_empirical,
                "empirical_pvalue_dist": self.allppermute}

    def __adjust(self, pv):
        pvadjust = np.array([min(pveff, 1) for pveff in (pv * self.adjustBy)])
        return pvadjust

    def __varianceDecomposition(self, cache=True):
        vd = limix.mtset.MTSet(Y=self.phenotypes, R=self.relatedness)
        vd_null_info = vd.fitNull(n_times=1000, rewrite=True)
        if vd_null_info['conv']:
            verboseprint("Variance decomposition converged")
            Cg = vd_null_info['Cg'] + 1e-4*sp.eye(self.P)
            Cn = vd_null_info['Cn'] + 1e-4*sp.eye(self.P)
            time = vd_null_info['time']
        else:
            sys.exit("Variance decomposition did not converge")

    ##############
    ### output ###
    ##############

    def manhattanQQ(self, pvalues, colorS='DarkBLue', colorNS='Orange',
            alphaNS=0.05, thr_plotting=0.05, savePlot=None):
        r"""

        Arguments:
            pvalues (array-like):
                [`P` x `NrSNP`] (single-trait) or [1 x `NrSNP`] (multi-trait) 
                array of p-values
            colorS (string):
                color of significant points
            colorNS (string):
                color of non-significant points
            alphaNS (float):
                plotting transparency of non-significant points
            thr_plotting (float):
                y-intercept for horizontal line as a marker for significance

        Returns:
            (None)
        """
        self.position, chromBounds = self.__getCumSum(self.genotypes_info)
        fig = plt.figure(figsize=[12,4])
        ax1 = fig.add_subplot(2,1,1)

        if self.fdr_empirical is not None:
            thr_plotting = self.fdr_empirical
        if self.mode is 'singletrait':
            pv = np.array(pvalues).min(axis=0).ravel()
        if self.mode is 'multitrait':
            pv = np.array(pvalues).ravel() 

        plot.plot_manhattan(posCum=self.position['pos_cum'].values.astype(int),
                pv=pv, colorS=colorS, colorNS=colorNS, alphaNS=alphaNS,
                thr_plotting=thr_plotting)

        ax1.set_title('%s' % self.chromosome)
        ax2 = fig.add_subplot(2,1,2)
        plot.qqplot(self.pvalues.ravel())
        fig.tight_layout()

        if saveTo is not None:
            fig.savefig('{}.png'.format(savePlot))

    def __getCumSum (self, offset=100000, chrom_len=None):
        RV = self.position.copy()
        chromvals = sp.unique(self.genotypes_info['chrom'])
        chrom_pos_cum = sp.zeros_like(chromvals)
        pos_cum = sp.zeros_like(self.genotypes_info.shape[0])
        offset = 100000
        if not 'pos_cum' in self.genotypes_info:
            RV["pos_cum"] = sp.zeros_like(self.genotypes_info['pos'])
            pos_cum = RV['pos_cum'].values
            maxpos_cum = 0
        for i, mychrom in enumerate(chromvals):
            chrom_pos_cum[i] = maxpos_cum
            i_chr=self.genotypes_info['chrom'] == mychrom
            if chrom_len is None:
                maxpos = self.genotypes_info['pos'][i_chr].values.astype(
                    int).max() + offset
            else:
                maxpos = chrom_len[i] + offset
                pos_cum[i_chr.values] = maxpos_cum + \
                        self.position.loc[i_chr,'pos'].values.astype(int)
                maxpos_cum += maxpos
        return (RV, chrom_pos_cum)
