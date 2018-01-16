###############
### modules ###
###############

import sys

# plotting
import argparse
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from matplotlib.backends.backend_pdf import PdfPages

# stats module
import scipy as sp
import pandas as pd
import numpy as np
import pylab as pl
from scipy_sugar.stats import quantile_gaussianize

# import LIMIX tools
import limix as limix

from limix.util.preprocess import regressOut
import limix.qtl as qtl
import limix.plot as plot
#from limix.stats.geno_summary import *

import limix.mtset

# import limmbo tools
from limmbo.utils.utils import verboseprint
from limmbo.utils.utils import effectiveTests

# other requirements
from math import sqrt
import copy
import pdb

#######################
### input functions ###
#######################

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
                sdfsdf
            setup (string):
                zdffds
            adjustSingleTrait (string):
                sdfdf

        Returns:
            (dictionary):
                dictionary containing:

                - **lm** (:class:`limix.qtl.LMM`):
                  LIMIX LMM object
                - **pvalues** (numpy array):
                  [`NrSNP` x `P`] (when mode is singletrait) or [1 x`NrSNP`]
                  array of p-values
                - **betas** (numpy array):
                  [`NrSNP` x `P`] array of effect size estimates per SNP across
                  all traits
                - **pvalues_adjust** (numpy array):
                  only returned if mode is 'singletrait' and 'adjustSingleTrait'
                  is not None; contains single-trait p-values adjusted for the
                  number of single-trait analyses conducted

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
            genotypes ():
            empiricalP (bool):
            computeFDR (bool):

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
        Single-trait association test. Wraps around `qtl.test_lmm`.

        Arguments:
            genotypes (array-like):
            [`N` x `NrSNP`] genotype array of [`N`] samples and [`NrSNP`]
                    genotypes

            empiricalP (bool):
            computeFDR (bool):
                adjustSingleTrait (string):
                    bonferroni or effective

        Returns:
            (dictionary):
                dictionary containing:

                - **lm**(:class:`limix.qtl.LMM`):
                  LIMIX LMM object
                - **pvalues** (numpy array):
                  [`NrSNP` x `P`] array of p-values
                - **betas** (numpy array):
                  [`P` x `NrSNP`] array of effect size estimates per SNP across
                  all traits
                - **pvalues_adjust** (numpy array):
                  only returned if 'adjustSingleTrait' is not None; contains
                  single-trait p-values adjusted for the number of single-trait
                  analyses conducted

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

        return {"lm":lm, "pvalues": pvalues, "pvalues_adjust": pvalues_adjust,
                "betas": betas}


    def saveAssociationResults(self, results):
        # from runAssociationAnalysis
        if self.options.permute:
            model = "%s_permute%s" % (model, self.seed)

        pvalues_out = self.writeResult(model=model, CHR=CHR, SNP=SNP, POS=POS)
        if self.options.noPlot is False:
            self.manhattanQQ(model=model, P=P)
            # from varianceDecomposition
        pd.DataFrame([time]).to_csv("%s/timeVarianceDecomposition_%s.csv" % 
                    (self.options.output, method), index=False, header=False)
        pd.DataFrame(Cg).to_csv("%s/Cg_%s.csv" % (self.options.output, method),
                    index=False, header=False)
        pd.DataFrame(Cn).to_csv("%s/Cn_%s.csv" % (self.options.output, method),
                    index=False, header=False)

        # from cpomute FDR
        outstring = (self.output, model, self.chromosome, self.fileend)
        pd.DataFrame(self.ppermute).to_csv("%s/%s_ppermute_%s%s.csv" %
                outstring, index=False)
        pd.DataFrame(['FDR', str(self.FDR)]).T.to_csv(
                "%s/%s_empiricalFDR_%s%s.csv" % outstring,
                header=False, index=False)
        return(Cg, Cn)

    def computeEmpiricalP(self, nrpermutations = 1000):
        r"""
        Compute empirical p-values: permute the genotypes, do the 
        association test, record if permuted p-value of SNP is smaller than
        original p-value. Sum these occurrences and divide by total number of
        permutation.

        Arguments:
            nrpermutations (int):
            number of permutations; 1/nrpermutations is the maximum level 
            of significance (alpha)to test for, 
            e.g nrpermuations=100 -> alpha=0.01 

        Returns:
        """

        verboseprint("Computing empirical p-values", verbose=self.verbose)
        np.random.seed(seed=self.seed)

        if self.mode is "multitrait":
            self.count = np.zeros((nrpermutations, self.S))
        else:
            self.count_raw = np.zeros((nrpermutations, self.P, self.S))

            self.pvalues_adjust_min = self.pvalues_adjust.min(axis=0)
            self.count = np.zeros((nrpermutations, self.S))

        for ps in range (nrpermutations):
            verboseprint("Permutation %s" % ps)
            genotypes_permute = self.genotypes[np.random.choice(self.N, self.N, 
                replace=False), :]
            if self.mode is "multitrait":
                resultsPermute = self.__multiTraitAssociation_anyeffect( 
                    genotypes = genotypes_permute, empiricalP=True)
                self.count[ps, :] = pvalues_permute <= self.pvalues
            if self.mode is "singletrait":
                resultsPermute = self.__singletraitAssociation(
                    genotypes = genotypes_permute, empiricalP=True)
                self.count_raw[ps,:, :] = pvalues_permute <= self.pvalues
                self.count[ps, :] = pvalues_permute_adjust.min(axis=0) <= \
                    self.pvalues_adjust_min

        self.pvalues_empirical = np.sum(self.count, axis=0)/nrpermutations
        if self.mode is "singletrait":
            self.pvalues_empirical_raw = np.sum(self.count_raw,
                    axis=0)/nrpermutations

    def __computeFDR(self):
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
                self.ppermute[ps, :] = resultsFDR['pvalues_permute']
            if self.mode == "singletrait":
                resultsFDR = self.__singleTraitAssociation(
                    genotypes = genotypes_permute, computeFDR=True)
                pvalues_adjust_min = resulsFDR['pvalues_permute_adjust'].min(
                    axis=0)
                self.ppermute[ps, :] = pvalues_adjust_min

        SNPsPassingFDR = int(self.fdr*self.S*tests)
        allppermute = self.ppermute.flatten()
        allppermute.sort()
        self.FDR = allppermute[SNPsPassingFDR]


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

    def manhattanQQ(self, model, colorS='DarkBLue', colorNS='Orange',
            alphaNS=0.05, thr_plotting=0.05):
        self.position, chromBounds = self.__getCumSum(self.genotypes_info)
        fig = plt.figure(figsize=[12,4])
        ax1 = fig.add_subplot(2,1,1)

        if self.options.computeFDR:
            thr_plotting = self.FDR
            self.fileend = "%s_%s%s" % (self.fileend, "FDR", self.fdr)
        if self.mode is 'singletrait':
            pv = self.pvalues_adjust.T.min(axis=1).ravel()
        if self.mode is 'multitrait':
            pv = self.pvalues.ravel() 

        plot.plot_manhattan(posCum=self.position['pos_cum'].values.astype(int),
                pv=pv, colorS=colorS, colorNS=colorNS, alphaNS=alphaNS,
                thr_plotting=thr_plotting)

        ax1.set_title('%s' % self.chromosome)
        ax2 = fig.add_subplot(2,1,2)
        plot.qqplot(self.pvalues.ravel())
        fig.tight_layout()
        fig.savefig('%s/%s_%s%s.png' % (self.output, self.chromosome, model,
            self.fileend))

    def __getCumSum (self, offset=100000, chrom_len=None):
        RV = self.position.copy()
            # sp.unique is always sorted
        chromvals = sp.unique(self.genotypes_info['chrom'])
            #get the starting position of each Chrom
        chrom_pos_cum = sp.zeros_like(chromvals)
        pos_cum = sp.zeros_like(self.genotypes_info.shape[0])
        offset = 100000
        if not 'pos_cum' in self.genotypes_info:
            #get the cum_pos of each variant.
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

    def __writeResult(self, model, CHR, SNP, POS, columns=None, thr=5e-8):
        outstring = (self.options.output, model, self.options.chromosome, 
                self.options.fileend)

        # getting SNP info
        verboseprint("extracting SNP info")
        SNP = np.array(self.genotypes_info.index)
        CHR = np.array(self.genotypes_info.iloc[:,:1])
        POS = np.array(self.genotypes_info.iloc[:,1:])
        if self.mode is 'singletrait':
            beta_df = pd.DataFrame(self.betas.T, index=SNP, columns=columns)
            stats_df = pd.DataFrame(self.stats.T, index=SNP, columns=columns)
            pvalue_df = pd.DataFrame(self.pvalues.T, index=SNP, 
                        columns=columns)
            pvalues_adjust_df = pd.DataFrame(self.pvalues_adjust.T, index=SNP, 
                        columns=columns)
            z_df = pd.DataFrame(self.z.T, index=SNP, columns=columns)
            pmin_df = pd.DataFrame(self.pvalues.T.min(axis=1), index=SNP, 
                        columns=['Pmin'])
            padjust_min_df = pd.DataFrame(self.pvalues_adjust.T.min(axis=1), 
                        index=SNP, columns=['Pmin'])
            

            pvalue_df['CHR'] = CHR
            pvalue_df['POS'] = POS
            pvalue_df['SNP'] = SNP
            
            pvalues_adjust_df['CHR'] = CHR
            pvalues_adjust_df['POS'] = POS
            pvalues_adjust_df['SNP'] = SNP

            pmin_df['CHR'] = CHR
            pmin_df['POS'] = POS
            pmin_df['SNP'] = SNP
                
            padjust_min_df['CHR'] = CHR
            padjust_min_df['POS'] = POS
            padjust_min_df['SNP'] = SNP
            
            beta_df['CHR'] = CHR
            beta_df['POS'] = POS
            beta_df['SNP'] = SNP

            stats_df['CHR'] = CHR
            stats_df['POS'] = POS
            stats_df['SNP'] = SNP

            z_df['CHR'] = CHR
            z_df['POS'] = POS
            z_df['SNP'] = SNP
            
            cols = pvalue_df.columns.tolist()
            cols = cols[len(cols)-3:len(cols)] + cols[:-3]

            beta_df = beta_df[cols]
            pvalue_df = pvalue_df[cols]
            pvalues_adjust_df = pvalues_adjust_df[cols]
            stats_df = stats_df[cols]
            z_df = z_df[cols]

            if self.pvalues_empirical is not None:
                cols_emp = cols[0:3]
                cols_emp.extend(['Pempirical'])
                
                pempirical_df = pd.DataFrame(self.pvalues_empirical.T,
                    index=SNP, columns=['Pempirical'])
                pempirical_df['CHR'] = CHR
                pempirical_df['POS'] = POS
                pempirical_df['SNP'] = SNP
                pempirical_df = pempirical_df[cols_emp]
                
                pempirical_raw_df = pd.DataFrame(self.pvalues_empirical_raw.T,
                    index=SNP, columns=columns)
                pempirical_raw_df['CHR'] = CHR
                pempirical_raw_df['POS'] = POS
                pempirical_raw_df['SNP'] = SNP
                pempirical_raw_df = pempirical_raw_df[cols]
            
                cols=cols[0:3]
                cols.extend(['Pmin'])
                pmin_df = pmin_df[cols]
                padjust_min_df = padjust_min_df[cols]
                psig_df = pmin_df.loc[pmin_df['Pmin'] < 5e-8]

            if pvalue_df.shape[1] !=4:
                pmin_df.to_csv("%s/%s_pminvalue_%s%s.csv" % outstring,
                    index=False)
                padjust_min_df.to_csv("%s/%s_padjust_minvalue_%s%s.csv" %
                    outstring, index=False)
                pvalues_adjust_df.to_csv("%s/%s_padjust_%s%s.csv" % outstring,
                    index=False)
                psig_df.to_csv("%s/%s_psigvalue_%s%s.csv" % outstring, index=False)
                stats_df.to_csv("%s/%s_statsvalue_%s%s.csv" % outstring,
                    index=False)
                z_df.to_csv("%s/%s_zvalue_%s%s.csv" % outstring, index=False)

        else:
            beta_df = pd.DataFrame(self.betas.T, index=SNP, columns=columns)
            pvalue_df = pd.DataFrame(SNP, index=SNP, columns=['SNP'])

            beta_df['SNP'] = SNP
            beta_df['CHR'] = CHR
            beta_df['POS'] = POS

            cols = beta_df.columns.tolist()
            cols = cols[len(cols)-3:len(cols)] + cols[:-3]

            beta_df = beta_df[cols]
            pvalue_df['CHR'] = CHR
            pvalue_df['POS'] = POS
            pvalue_df['P'] = self.pvalues.flatten()

        if self.pvalues_empirical is not None:
            pempirical_df = pd.DataFrame(SNP, index=SNP, columns=['SNP'])
            pempirical_df['CHR'] = CHR
            pempirical_df['POS'] = POS
            pempirical_df['P'] = self.pvalues_empirical

        if self.options.likelihoods:
            maNLL0_df = pd.DataFrame(SNP, index=SNP, columns=['SNP'])
            NLL0_df['CHR'] = CHR
            NLL0_df['POS'] = POS
            NLL0_df['NLL'] = self.NLL0.T
            NLLAlt_df = pd.DataFrame(SNP, index=SNP, columns=['SNP'])
            NLLAlt_df['CHR'] = CHR
            NLLAlt_df['POS'] = POS
            NLLAlt_df['NLL'] = self.NLLAlt.T
            
        pvalue_df.to_csv("%s/%s_pvalue_%s%s.csv" % outstring, index=False)
        beta_df.to_csv("%s/%s_betavalue_%s%s.csv" % outstring, index=False)

        if self.pvalues_empirical is not None:
            pempirical_df.to_csv("%s/%s_pempirical_%s%s%s.csv" %
                (outstring + (self.fdr,)), index=False)
            
        if self.pvalues_empirical_raw is not None:
            pempirical_raw_df.to_csv("%s/%s_pempirical_raw%s%s%s.csv" %
                (outstring + (self.fdr,)), index=False)
        if self.likelihoods:
            NLL0_df.to_csv("%s/%s_NLL0_%s%s.csv" % outstring, index=False)
            NLLAlt_df.to_csv("%s/%s_NLLAlt_%s%s.csv" % outstring, index=False)
