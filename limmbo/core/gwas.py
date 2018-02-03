import sys

import scipy as sp
import pandas as pd
import numpy as np

import limix.qtl as qtl
import limix.plot as plot
import limmbo.plot.manhattan as mh
import limix.mtset

from limmbo.utils.utils import verboseprint
from limmbo.utils.utils import effectiveTests

import matplotlib as mpl
mpl.use('Agg')


class GWAS(object):
    r"""
    Class to run association tests.

    Arguments:
        datainput (:class:`limmbo.io.InputData`):
           Object containing relevant data for association study,
           at least phenotypes and genotypes matrix.
        verbose (bool):
            Set to true to print progress messages.

    """
    def __init__(self, datainput, verbose=True):
        self.verbose = verbose
        self.seed = None
        self.searchDelta = False
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
        self.model = None
        self.pvalues_empirical = None
        self.adjustBy = None
        self.estimate_vd = None
        self.fdr_empirical = None

        try:
            self.genotypes = np.array(self.genotypes)
        except:
            raise IOError("datainput.genotypes cannot be coverted to np.array")

        try:
            self.phenotypes = np.array(self.phenotypes)
        except:
            raise IOError(
                "datainput.phenotypes cannot be coverted to np.array")

        if self.covariates is not None:
            try:
                self.covariates = np.array(self.covariates)
            except:
                raise IOError(
                    "datainput.covariates cannot be coverted to np.array")

        if self.relatedness is not None:
            try:
                self.relatedness = np.array(self.relatedness)
            except:
                raise IOError(
                    "datainput.relatedness cannot be coverted to np.array")

    def runAssociationAnalysis(self, mode, setup="lmm",
                               adjustSingleTrait=None):
        r"""
        Analysing the association between phenotypes, genotypes, optional
        covariates and random genetic effects.

        Arguments:
            mode (string):
                pecifies the type of linear model: either 'multitrait' for
                multivariate analysis or 'singletrait' for univariate analysis.
            setup (string):
                specifies the linear model setup: either 'lmm' for linear mixed
                model or 'lm' for a simple linear model.
            adjustSingleTrait (string):
                Method to adjust single-trait association p-values for testing
                multiple traits; If None (default) no adjusting. Options are
                'bonferroni' (for bonferroni correction') or 'effective' (for
                correcting for the effective number of tests as described in
                `(Galwey,2009) <http://onlinelibrary.wiley.com/doi/10.
                1002/gepi.20408/abstract>`_.


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
                  only returned if mode is 'singletrait' and
                  'adjustSingleTrait' is not None; contains single-trait
                  p-values adjusted for the number of single-trait analyses
                  conducted.

        Examples:

            .. doctest::

                >>> from pkg_resources import resource_filename
                >>> from limmbo.io.reader import ReadData
                >>> from limmbo.io.input import InputData
                >>> from limmbo.core.gwas import GWAS
                >>> data = ReadData(verbose=False)
                >>> file_pheno = resource_filename('limmbo',
                ...                                'io/test/data/pheno.csv')
                >>> file_geno = resource_filename(
                ...     'limmbo', 'io/test/data/genotypes.csv')
                >>> file_relatedness = resource_filename('limmbo',
                ...                     'io/test/data/relatedness.csv')
                >>> file_covs = resource_filename('limmbo',
                ...                               'io/test/data/covs.csv')
                >>> file_Cg = resource_filename('limmbo',
                ...                     'io/test/data/Cg.csv')
                >>> file_Cn = resource_filename('limmbo',
                ...                     'io/test/data/Cn.csv')
                >>> data.getPhenotypes(file_pheno=file_pheno)
                >>> data.getCovariates(file_covariates=file_covs)
                >>> data.getRelatedness(file_relatedness=file_relatedness)
                >>> data.getGenotypes(file_genotypes=file_geno)
                >>> data.getVarianceComponents(file_Cg=file_Cg,
                ...                            file_Cn=file_Cn)
                >>> indata = InputData(verbose=False)
                >>> indata.addPhenotypes(phenotypes = data.phenotypes)
                >>> indata.addRelatedness(relatedness = data.relatedness)
                >>> indata.addCovariates(covariates = data.covariates)
                >>> indata.addGenotypes(genotypes=data.genotypes,
                ...                     genotypes_info=data.genotypes_info)
                >>> indata.addVarianceComponents(Cg = data.Cg, Cn=data.Cn)
                >>> indata.commonSamples()
                >>> indata.regress()
                >>> indata.transform(transform="scale")
                >>> gwas = GWAS(datainput=indata, verbose=False)
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
                '4.584e-09'
                >>> resultsAssociation['betas'].shape
                (10, 20)
                >>>
                >>>
                >>> # Example of single-trait single-variant association
                >>> # testing using a linear mixed model.
                >>> resultsAssociation = gwas.runAssociationAnalysis(
                ...     setup="lmm", mode="singletrait",
                ...     adjustSingleTrait = "effective")
                >>> resultsAssociation.keys()
                ['pvalues_adjust', 'lm', 'betas', 'pvalues']
                >>> resultsAssociation['pvalues'].shape
                (10, 20)
                >>> resultsAssociation['betas'].shape
                (10, 20)
                >>> '{:0.3e}'.format(
                ...     resultsAssociation['pvalues_adjust'].min())
                '2.262e-03'
        """

        # set parameters for the analysis
        self.N, self.P = self.phenotypes.shape
        self.S = self.genotypes.shape[1]
        verboseprint("Loaded {} samples, {} phenotypes, {} snps".format(
            self.N, self.P, self.S), verbose=self.verbose)

        self.setup = setup
        self.mode = mode

        if mode == "multitrait":
            associationResults = self.__multiTraitAssociation_anyeffect(
                genotypes=self.genotypes)

        if mode == "singletrait":
            associationResults = self.__singleTraitAssociation(
                genotypes=self.genotypes,
                adjustSingleTrait=adjustSingleTrait)

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
                if self.P > 30:
                    print("Warning: For large trait sizes, computation times "
                          "for pure REML variance decomposition are long, "
                          "consider bootstrapping trait-trait covariance"
                          " components")
                verboseprint("Estimate Variance components",
                             verbose=self.verbose)
                self.__varianceDecomposition()
            K1c = self.Cg
            K2c = self.Cn
            K1r = self.relatedness
            self.model = "lmm_mt"
        else:
            K1c = 1e-9 * sp.eye(self.P)
            K2c = sp.cov(self.phenotypes.T)
            K1r = sp.eye(self.N)

            if self.pcs is not None:
                self.model = "lm_mt_pcs"
            else:
                self.model = "lm_mt"
        Asnps = sp.eye(self.P)
        if not empiricalP and not computeFDR:
            verboseprint("Computing multi-trait (any effect) model: {}".format(
                self.model), verbose=self.verbose)

        lm, pvalues = qtl.qtl_test_lmm_kronecker(snps=genotypes,
                                                 phenos=self.phenotypes,
                                                 Asnps=Asnps, Acovs=Acovs,
                                                 covs=self.covariates, K1r=K1r,
                                                 K1c=K1c, K2c=K2c,
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
                `(Galwey,2009) <http://onlinelibrary.wiley.com/doi/10.
                1002/gepi.20408/abstract>`_.

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
                      only returned if 'adjustSingleTrait' is not None;
                      contains single-trait p-values adjusted for the number of
                      single-trait analyses conducted.

        """

        if self.setup is "lmm":
            self.model = "lmm_st"
            K = self.relatedness
        else:
            if self.pcs is not None:
                self.model = "lm_st_pcs"
            else:
                self.model = "lm_st"
                K = None

        if not empiricalP and not computeFDR:
            verboseprint("Computing single-trait association ({})".format(
                self.model), verbose=self.verbose)
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
                    raise (("{} is not a provided method to adjust "
                            "single-trait pvalues for multiple hypothesis "
                            "testing").format(
                        adjustSingleTrait))
            pvalues_adjust = np.array([self.__adjust(p) for p in pvalues])
        else:
            pvalues_adjust = None

        self.adjustSingleTrait = adjustSingleTrait

        return {"lm": lm, "pvalues": pvalues, "pvalues_adjust": pvalues_adjust,
                "betas": betas}

    def saveAssociationResults(self, results, outdir, name="",
                               pvalues_empirical=None):
        r"""
        Saves results of association analyses.

        Arguments:
            results (dictionary):
                dictionary generated via runAssociation analysis, containing:

                - **lm** (:class:`limix.qtl.LMM`):
                  LIMIX LMM object
                - **pvalues** (numpy array):
                  [`P` x `NrSNP`] array of p-values
                - **betas** (numpy array):
                  [`P` x `NrSNP`] array of effect size estimates per SNP
                  across all traits
                - **pvalues_adjust** (numpy array):
                  only returned mode == singletrait and if
                  'self.adjustSingleTrait' is not None; contains single-trait
                  p-values adjusted for the number of single-trait analyses
                  conducted.

            outdir (string):
                '/path/to/output/directory'; needs user writing permission
            name (string):
                input name specific name, such as chromosome used in analysis
            pvalues_empirical (pandas dataframe, optional):
                empirical pvalues via adjustSingleTrait.

        Returns:
            (None)
        """
        if name is not "":
            self.name = name
            outstring = (outdir, self.model, "_" + name)
        else:
            self.name = ""
            outstring = (outdir, self.model, "")

        self.genotypes_info['SNP'] = self.genotypes_info.index

        beta_df = pd.DataFrame(results['betas'].T,
                               index=self.genotypes_info.index,
                               columns=self.phenotype_ID)
        beta_df = pd.concat([self.genotypes_info, beta_df], axis=1)

        if self.mode is 'singletrait':
            pvalues_df = pd.DataFrame(results['pvalues'].T,
                                      index=self.genotypes_info.index,
                                      columns=self.phenotype_ID)
            pvalues_df = pd.concat([self.genotypes_info, pvalues_df], axis=1)

            if results['pvalues_adjust'] is not None:
                pvalues_adjust_df = pd.DataFrame(
                    results['pvalues_adjust'].T,
                    index=self.genotypes_info.index,
                    columns=self.phenotype_ID)
                pvalues_adjust_df = pd.concat([self.genotypes_info,
                                               pvalues_adjust_df], axis=1)

                pvalues_adjust_df.to_csv("%s/%s_padjust%s.csv" % outstring,
                                         index=False)

            if pvalues_empirical is not None:

                pempirical_df = pd.DataFrame(pvalues_empirical.T,
                                             index=self.genotypes_info.index,
                                             columns=self.phenotype_ID)
                pempirical_df = pd.concat([self.genotypes_info,
                                           pempirical_df], axis=1)

                pempirical_raw_df = pd.DataFrame(
                    self.pvalues_empirical_raw.T,
                    index=self.genotypes_info.index,
                    columns=self.phenotype_ID)
                pempirical_raw_df = pd.concat([self.genotypes_info,
                                               pempirical_raw_df], axis=1)

                pempirical_raw_df.to_csv("%s/%s_pempirical_raw%s%s.csv" %
                                         (outstring + (self.nrpermutations,)),
                                         index=False)

        if self.mode is 'multitrait':
            pvalues_df = pd.DataFrame(results['pvalues'].T,
                                      index=self.genotypes_info.index,
                                      columns=["P"])
            pvalues_df = pd.concat([self.genotypes_info, pvalues_df], axis=1)

            if pvalues_empirical is not None:
                pvalues_empirical_df = pd.DataFrame(
                    pvalues_empirical.T, index=self.genotypes_info.index,
                    columns=["P"])
                pvalues_empirical_df = pd.concat(
                    [self.genotypes_info, pvalues_empirical_df], axis=1)

        pvalues_df.to_csv("%s/%s_pvalue%s.csv" % outstring, index=False)
        beta_df.to_csv("%s/%s_betavalue%s.csv" % outstring, index=False)

        if pvalues_empirical is not None:
            pvalues_empirical_df.to_csv("%s/%s_pempirical%s%s.csv" %
                                        (outstring + (self.nrpermutations,)),
                                        index=False)

        if self.estimate_vd:
            if self.timeVD is not None:
                pd.DataFrame(self.timeVD).to_csv(
                    "%s/timeVarianceDecomposition_REML.csv" % outdir,
                    index=False, header=False)

            pd.DataFrame(self.Cg).to_csv("%s/Cg_REML.csv" % (outdir),
                                         index=False, header=False)
            pd.DataFrame(self.Cn).to_csv("%s/Cn_REML.csv" % (outdir),
                                         index=False, header=False)

        if self.fdr_empirical is not None:
            pd.DataFrame(self.allppermute).to_csv("%s/%s_ppermute%s.csv" %
                                                  outstring, index=False)
            pd.DataFrame(['FDR', str(self.fdr_empirical)]).T.to_csv(
                "%s/%s_empiricalFDR%s.csv" % outstring, header=False,
                index=False)

    def computeEmpiricalP(self, pvalues, nrpermutations=1000, seed=10):
        r"""
        Compute empirical p-values: permute the genotypes, do the
        association test, record if permuted p-value of SNP is smaller than
        original p-value. Sum these occurrences and divide by total number of
        permutation.

        Arguments:
            pvalues (array-like):
                [`P` x `NrSNP`] (single-trait) or [1 x `NrSNP`] (multi-trait)
                array of p-values.
            nrpermutations (int):
                number of permutations; 1/nrpermutations is the maximum level
                of significance (alpha)to test for, e.g.
                nrpermuations=100 -> alpha=0.01
            seed (int):
                Seed to initiate random number generator

        Returns:
            (numpy array):
                [`P` x `NrSNP`] (single-trait) or [1 x `NrSNP`] (multi-trait)
                array of empirical p-values
        """

        verboseprint("Computing empirical p-values", verbose=self.verbose)

        self.seed = seed
        np.random.seed(seed=self.seed)

        self.nrpermutations = nrpermutations
        pvalues = np.array(pvalues)

        if self.mode is "multitrait":
            count = np.zeros((nrpermutations, self.S))
        else:
            count = np.zeros((nrpermutations, self.P, self.S))

        for ps in range(nrpermutations):
            verboseprint("Permutation %s" % (ps+1))
            genotypes_permute = self.genotypes[np.random.choice(
                self.N, self.N, replace=False), :]
            if self.mode is "multitrait":
                resultsPermute = self.__multiTraitAssociation_anyeffect(
                    genotypes=genotypes_permute, empiricalP=True)
                count[ps, :] = resultsPermute['pvalues'] <= pvalues
            if self.mode is "singletrait":
                resultsPermute = self.__singletraitAssociation(
                    genotypes=genotypes_permute, empiricalP=True)
                count[ps, :, :] = resultsPermute['pvalues'] <= pvalues

        pvalues_empirical = np.sum(count, axis=0) / nrpermutations

        return pvalues_empirical

    def computeFDR(self, fdr, seed=10):
        r"""
        Create an empirical p-values distribution by permuting the genotypes
        and running the association test on these (10 random permuations).
        Recored the observed pvalues, order by rank and find the rank of the
        desired FDR to determine the empirical FDR.

        Arguments:
            fdr (float):
                desired fdr threshold
            seed (int):
                Seed to initiate random number generator

        Returns:
            (tuple):
                tuple containing:

                    - **empirical fdr** (float):
                    - **empirical_pvalue_distribution** (numpy array):
                      array of empirical p-values for 10 permutations tests
        """

        verboseprint("Computing empirical FDR", verbose=self.verbose)
        self.seed = seed
        np.random.seed(seed=self.seed)
        # tests = int(1/self.options.fdr)
        tests = 10

        ppermute = np.zeros((tests, self.S))

        for ps in range(tests):
            genotypes_permute = self.genotypes[np.random.choice(
                self.N, self.N, replace=False), :]
            if self.mode == "multitrait":
                resultsFDR = self.__multiTraitAssociation_anyeffect(
                    genotypes=genotypes_permute, computeFDR=True)
                ppermute[ps, :] = resultsFDR['pvalues']
            if self.mode == "singletrait":
                resultsFDR = self.__singleTraitAssociation(
                    genotypes=genotypes_permute, computeFDR=True,
                    adjustSingleTrait=self.adjustSingleTrait)
                pvalues_adjust_min = resultsFDR['pvalues_adjust'].min(
                    axis=0)
                ppermute[ps, :] = pvalues_adjust_min

        SNPsPassingFDR = int(fdr * self.S * tests)
        self.allppermute = ppermute.flatten()
        self.allppermute.sort()
        self.fdr_empirical = self.allppermute[SNPsPassingFDR]

        return self.fdr_empirical, self.allppermute

    def __adjust(self, pv):
        r"""
        Adjust pvalues for multiple hypothesis testing by multiplying with
        a constant (if adjustBy=P, equivalent to Bonferroni adjustment).

        Arguments:
            pv (array like):
                p-values

        Returns:
            pv (numpy array):
                adjusted p-values
        """
        pvadjust = np.array([min(pveff, 1) for pveff in (pv * self.adjustBy)])
        return pvadjust

    def __varianceDecomposition(self, cache=True):
        vd = limix.mtset.MTSet(Y=self.phenotypes, R=self.relatedness)
        vd_null_info = vd.fitNull(n_times=1000, rewrite=True)
        if vd_null_info['conv']:
            verboseprint("Variance decomposition converged")
            self.Cg = vd_null_info['Cg'] + 1e-4 * sp.eye(self.P)
            self.Cn = vd_null_info['Cn'] + 1e-4 * sp.eye(self.P)
            self.time = vd_null_info['time']
        else:
            sys.exit("Variance decomposition did not converge")

    def manhattanQQ(self, results, colourS='DarkBLue', colourNS='Orange',
                    alphaS=1, alphaNS=0.1, thr_plotting=None, saveTo=None):
        r"""
        Plot manhattan and quantile-quantile plot of association results.

        Arguments:
            results (dictionary):
                dictionary generated via runAssociation analysis
            colourS (string, optional):
                colour of significant points
            colourNS (string, optional):
                colour of non-significant points
            alphaS (float, optional):
                plotting transparency of significant points
            alphaNS (float, optional):
                plotting transparency of non-significant points
            thr_plotting (float, optional):
                y-intercept for horizontal line as a marker for significance
            saveTo (string, optional):
                /path/to/output/directory to automatically save plot as pdf;
                needs user writing permission.

        Returns:
            (None)
        """
        import matplotlib.pyplot as plt
        if self.mode is 'singletrait':
            fig = plt.figure(figsize=[12, 4])
            pv_min = np.array(results['pvalues']).min(axis=0)
            data = pd.DataFrame({'pv': pv_min,
                                 'chrom': np.array(self.genotypes_info['chrom']
                                                   ).astype('int'),
                                 'pos': np.array(self.genotypes_info['pos']
                                                 ).astype('int')})
            ax1 = fig.add_subplot(2, 1, 1)
            mh.plot_manhattan(df=data,
                              null_style={'alpha': float(
                                  alphaNS), 'color': colourNS},
                              alt_style={'alpha': float(
                                  alphaS), 'color': colourS},
                              alpha=thr_plotting)
            ax1.set_title('%s' % self.name)
            ax2 = fig.add_subplot(2, 1, 2)
            plot.qqplot(results['pvalues'].ravel(), alphaLevel=None)
            fig.tight_layout()

            if results['pvalues_adjust'] is not None:
                fig = plt.figure(figsize=[12, 8])
                pv_adjust_min = np.array(results['pvalues_adjust']).min(
                    axis=0).ravel()
                data_adjust = pd.DataFrame({'pv': pv_adjust_min,
                                            'chrom': np.array(
                                                self.genotypes_info['chrom']
                                            ).astype('int'),
                                            'pos': np.array(
                                                self.genotypes_info['pos']
                                            ).astype('int')})
                ax1 = fig.add_subplot(4, 1, 1)
                mh.plot_manhattan(df=data,
                                  null_style={'alpha': float(
                                      alphaNS), 'color': colourNS},
                                  alt_style={'alpha': float(
                                      alphaS), 'color': colourS},
                                  alpha=thr_plotting)
                ax1.set_title('%s (p-values)' % self.name)
                ax2 = fig.add_subplot(4, 1, 2)
                plot.qqplot(results['pvalues'].ravel())
                ax1 = fig.add_subplot(4, 1, 3)
                mh.plot_manhattan(df=data_adjust,
                                  null_style={'alpha': float(
                                      alphaNS), 'color': colourNS},
                                  alt_style={'alpha': float(
                                      alphaS), 'color': colourS},
                                  alpha=thr_plotting)
                ax1.set_title('%s (p-values adjust)' % self.name)
                ax2 = fig.add_subplot(4, 1, 4)
                plot.qqplot(results['pvalues_adjust'].ravel())
                fig.tight_layout()

        if self.mode is 'multitrait':
            pv = np.array(results['pvalues']).ravel()
            data = pd.DataFrame({'pv': pv,
                                 'chrom': np.array(self.genotypes_info['chrom']
                                                   ).astype('int'),
                                 'pos': np.array(self.genotypes_info['pos']
                                                 ).astype('int')})
            fig = plt.figure(figsize=[12, 4])
            ax1 = fig.add_subplot(2, 1, 1)
            mh.plot_manhattan(df=data,
                              null_style={'alpha': float(
                                  alphaNS), 'color': colourNS},
                              alt_style={'alpha': float(
                                  alphaS), 'color': colourS},
                              alpha=thr_plotting)
            ax1.set_title('%s' % self.name)
            ax2 = fig.add_subplot(2, 1, 2)
            plot.qqplot(results['pvalues'].ravel())
            fig.tight_layout()

        if saveTo is not None:
            if self.name is not "":
                output = '{}/{}_{}.png'.format(saveTo, self.name, self.model)
            else:
                output = '{}/{}.png'.format(saveTo, self.model)
            fig.savefig(output)
