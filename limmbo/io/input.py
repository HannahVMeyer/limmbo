###############
### modules ###
###############

import h5py

import sys
sys.path.append('./../../')
#sys.path.append('/homes/hannah/bin/python_modules')
#sys.path.append(
   # '/nfs/gns/homes/hannah/software/python2.7.8/lib/python2.7/site-packages')


import scipy as sp
import pandas as pd
import numpy as np
import re

# import LIMIX tools
import limix
import limix.io.genotype_reader as gr

# import mtSet tools
from mtSet.pycore.utils.normalization import gaussianize
from mtSet.pycore.utils.normalization import regressOut

# other requirements
from math import sqrt
from limmbo.utils.utils import verboseprint
from limmbo.utils.utils import match

########################
### functions: input ###
########################

class DataInput(object):
    """
    """

    def __init__(self, options=None):
        '''
        nothing to initialize
        '''
        self.options = options
        self.samples = None
        self.phenotypes = None
        self.pheno_samples = None
        self.phenotype_ID = None
        self.snps = None
        self.position = None
        self.geno_samples = None
        self.covariates = None
        self.covariate_samples = None
        self.relatedness = None
        self.relatedness_samples = None
        self.pcs = None
        self.pc_samples = None
        self.Cg = None
        self.Cn = None
        self.trainset = None
        self.traitsarray = None

    def getPhenotypes(self):
        verboseprint("Extracting phenotypes", verbose=self.options.verbose)
        if re.search(".h5", self.options.file_pheno) is None:
            self.phenotypes = pd.io.parsers.read_csv(
                self.options.file_pheno, index_col=0)
            self.phenotype_ID = np.array(self.phenotypes.columns)
            self.pheno_samples = np.array(self.phenotypes.index)
            self.phenotypes = np.array(self.phenotypes)
        else:
            file = h5py.File(self.options.file_pheno, 'r')
            self.phenotypes = file['phenotype']['matrix'][:]
            self.phenotype_ID = np.array(
                file['phenotype']['col_header']['phenotype_ID'][:].astype(
                    'str'))
            self.pheno_samples = np.array(
                file['phenotype']['row_header']['sample_ID'][:].astype('str'))
            self.phenotypes = np.array(self.phenotypes)
            file.close()
        return self

    def getGenotypes(self):
        if re.search(".h5", self.options.file_geno) is None:
            verboseprint("Extracting genotypes from .csv file",
                         verbose=self.options.verbose)
            genotypes = pd.io.parsers.read_csv(
                self.options.file_geno, index_col=0, header=0)
            snp_info = np.array(genotypes.index)

            position = []
            snp_ID = []
            for id in range(snp_info.shape[0]):
                split = np.array(snp_info[id].split('-'))
                snp_ID.append(split[2])
                position.append(split[[0, 1]])

            self.geno_samples = np.array(genotypes.columns)
            self.snps = np.array(genotypes).astype(float).T
            self.position = pd.DataFrame(np.array(position), columns=[
                                         'chrom', 'pos'], index=snp_ID)
        else:
            # read genotype information: chromosome-wise hf5 files
            geno_reader = gr.genotype_reader_h5py(self.options.file_geno)
            verboseprint("Extracting genotypes from hf5 file",
                         verbose=self.options.verbose)
            self.geno_samples = geno_reader.sample_ID
            self.snps = geno_reader.getGenotypes().astype(float)
            self.position = geno_reader.getPos()

        if self.options.standardise:
            verboseprint("Standardise genotypes", verbose=self.options.verbose)
            self.standardiseGenotypes()
        return self

    def getCovariates(self):
        if self.options.file_covariates != None:
            verboseprint("Reading covariates file",
                         verbose=self.options.verbose)
            self.covariates = pd.io.parsers.read_csv(
                self.options.file_covariates)
            self.covs_samples = np.ravel(self.covariates.iloc[:, :1])
            self.covariates = np.array(
                self.covariates.iloc[:, 1:]).astype(float)
            # Bug in LIMIX: this concatenation should be done in QTL function;
            # adjusts for mean of covariates
            if not self.options.regress:
                self.covariates = sp.concatenate(
                    [self.covariates,
                        sp.ones((self.covariates.shape[0], 1))], 1)
                self.covariates = np.array(self.covariates)
        else:
            # When cov are not set (None), LIMIX considers an intercept
            verboseprint("No covariates set", verbose=self.options.verbose)
            self.covariates = None
            self.covs_samples = None
        return self

    def getKinship(self):
        if self.options.file_kinship != None:
            verboseprint("Reading relationship matrix",
                         verbose=self.options.verbose)
            self.relatedness = pd.io.parsers.read_csv(
                self.options.file_kinship)
            self.relatedness_samples = np.array(self.relatedness.columns)
            self.relatedness = np.array(self.relatedness).astype(float)
        else:
            self.relatedness = None
            verboseprint("No relationship matrix set",
                         verbose=self.options.verbose)
            self.relatedness_samples = None
        return self

    def getPCs(self):
        if self.options.file_pcs != None:
            verboseprint("Reading PCs", verbose=self.options.verbose)
            self.pcs = pd.io.parsers.read_csv(
                self.options.file_pcs, header=None, sep=" ")
            self.pc_samples = np.array(self.pcs.iloc[:, :1]).flatten()
            self.pcs = np.array(self.pcs.iloc[:, 2:]).astype(float)
            verboseprint("Extracting first %s pcs" %
                         self.options.nrpcs, verbose=self.options.verbose)
            self.pcs = self.pcs[:, :self.options.nrpcs]
        else:
            self.pcs = None
            verboseprint("No pcs set", verbose=self.options.verbose)
            self.pc_samples = None
        return self

    def getVD(self):
        if self.options.mode == 'multitrait':
            if self.options.file_Cg is None and self.options.file_Cn is None:
                verboseprint(("No variance components supplied, run VD/limmbo"
                              "before lmm test"), verbose=self.options.verbose)
                self.Cg, self.Cn = None, None
            elif self.options.file_Cg is None or self.options.file_Cn is None:
                verboseprint(("Both variant components need to be supplied: Cg"
                              "is %s and Cn is %s") % (self.options.file_Cg,
                                                       self.options.file_Cn),
                             verbose=self.options.verbose)
                self.Cg, self.Cn = None, None
            else:
                self.Cg = np.array(pd.io.parsers.read_csv(
                    self.options.file_Cg, header=None))
                self.Cn = np.array(pd.io.parsers.read_csv(
                    self.options.file_Cn, header=None))
            return self

    def standardiseGenotypes(self):
        for snp in range(self.snps.shape[1]):
            p, q = AlleleFrequencies(self.snps[:, snp])
            var_snp = sqrt(2 * p * q)
            for n in range(self.snps[:, snp].shape[0]):
                self.snps[n, snp] = (self.snps[n, snp] - 2 * q) / var_snp
        return self

    def getAlleleFrequencies(self):
        verboseprint("Get allele frequencies of %s snps from chromosome %s" % (
            self.snps.shape[1], self.options.chromosome))
        self.freqs = np.zeros((self.snps.shape[1], 3))
        for snp in range(self.snps.shape[1]):
            self.freqs[snp, 1], self.freqs[snp, 2] = AlleleFrequencies(
                self.snps[:, snp])
        self.freqs = self.freqs.astype('str')
        self.freqs[:, 0] = np.array(self.position.index)

        pd.DataFrame(self.freqs, columns=["SNP_ID", "A1", "A2"]).to_csv(
            "%s/allelefrequencies_%s.csv" % (self.options.output,
                                             self.options.chromosome),
            index=False, header=True)
        return self

    def subsetTraits(self):
        if self.options.traitstring is not None:
            verboseprint("Chose subset of %s traits" %
                         self.options.traitstring,
                         verbose=self.options.verbose)
            traitslist = [x.split('-')
                          for x in self.options.traitstring.split(',')]
            self.traitsarray = []
            for t in traitslist:
                if len(t) == 1:
                    self.traitsarray.append(int(t[0]) - 1)
                else:
                    [self.traitsarray.append(x) for x in range(
                        int(t[0]) - 1, int(t[1]))]

            self.phenotypes = self.phenotypes[:, self.traitsarray]
            self.phenotype_ID = self.phenotype_ID[self.traitsarray]
            if self.Cg is not None:
                self.Cg = self.Cg[self.traitsarray, :][:, self.traitsarray]
                self.Cn = self.Cn[self.traitsarray, :][:, self.traitsarray]
        return self

    def commonSamples(self):
        if self.options.file_geno is not None:
            if self.options.file_samplelist is not None:
                verboseprint(("Read sample list to be extracted from"
                              "phenotypes (samples:%s) and genotypes (samples:"
                              "%s)") % (
                    len(self.pheno_samples), len(self.geno_samples)),
                    verbose=self.options.verbose)
                # read sample list
                subset = np.array(pd.io.parsers.read_csv(
                    self.options.file_samplelist, header=None))
                verboseprint("Number of samples in sample list: %s" %
                             len(subset))
            else:
                verboseprint(("Get common samples between phenotypes"
                              "(samples:%s) and genotypes (samples: %s)") % (
                    len(self.pheno_samples), len(self.geno_samples)),
                    verbose=self.options.verbose)
                # get common samples between genotypes and phenotypes
                subset = np.intersect1d(self.pheno_samples, self.geno_samples)
                verboseprint(("Number of common samples between phenotypes and"
                              "genotypes: %s") % len(
                    subset), verbose=self.options.verbose)

            # subsample arrays and match order of phenotypes/covariates/kinship
            # and respective samples to genotypes
            verboseprint(("Match order of pheno_samples to geno_samples and"
                          "extract corresponding samples in right order from"
                          "additional optional files (kinship, covariates,"
                          "pcs)"),
                         verbose=self.options.verbose)
            subset_geno_samples = np.in1d(self.geno_samples, subset)
            self.geno_samples = self.geno_samples[subset_geno_samples]
            self.snps = self.snps[subset_geno_samples, :]
            self.samples = self.geno_samples
            if self.options.permute is True:
                verboseprint("Permuting genotype samples (seed %s)" %
                             self.options.seed, verbose=self.options.verbose)
                self.snps = self.snps[np.random.RandomState(
                    self.options.seed).choice(self.snps.shape[0],
                                              self.snps.shape[0],
                                              replace=False), :]

            subset_pheno_samples = np.in1d(self.pheno_samples, subset)
            self.pheno_samples = self.pheno_samples[subset_pheno_samples]
            self.phenotypes = self.phenotypes[subset_pheno_samples, :]
            self.phenotypes, self.pheno_samples, samples_before,
            samples_after = match(
                self.geno_samples, self.pheno_samples, self.phenotypes,
                squarematrix=False)
        else:
            subset = self.pheno_samples
            self.samples = self.pheno_samples

        if self.relatedness is not None:
            subset_relatedness_samples = np.in1d(
                self.relatedness_samples, subset)
            self.relatedness_samples = self.relatedness_samples[
                subset_relatedness_samples]
            self.relatedness = self.relatedness[subset_relatedness_samples, :]
            (self.relatedness, self.relatedness_samples, samples_before,
             samples_after) = match(
                self.samples, self.relatedness_samples, self.relatedness,
                squarematrix=True)

        if self.covariates is not None:
            subset_covs_samples = np.in1d(self.covs_samples, subset)
            self.covs_samples = self.covs_samples[subset_covs_samples]
            self.covariatess = self.covariates[subset_covs_samples, :]
            self.covariates, self.covs_samples, samples_before,
            samples_after = match(
                self.samples, self.covs_samples, self.covariates,
                squarematrix=False)

        if self.pcs is not None:
            subset_pc_samples = np.in1d(self.pc_samples, subset)
            self.pc_samples = self.pc_samples[subset_pc_samples]
            self.pcs = self.pcs[subset_pc_samples, :]
            self.pcs, self.pc_samples, samples_before, samples_after = match(
                self.samples, self.pc_samples, self.pcs, squarematrix=False)

    def regress_and_transform(self):
        if self.options.regress:
            type = "covariates"
            if self.pcs is not None:
                if self.covariates is not None:
                    verboseprint("Append Pcs to covariates",
                                 verbose=self.options.verbose)
                    self.covariates = sp.concatenate(
                        [self.covariates, self.pcs], axis=1)
                    type = "covariates and PCs"
                else:
                    self.covariates = self.pcs
                    type = "PCs"
            verboseprint("Regress out %s" %
                         type,  verbose=self.options.verbose)
            self.phenotypes = regressOut(self.phenotypes, self.covariates)
            self.covariates = None
            self.transform()
        return self

    def transform(self):
        if self.options.transform == "scale":
            verboseprint("Use %s as transformation" %
                         self.options.transform, verbose=self.options.verbose)
            self.phenotypes = scale(self.phenotypes)
        elif self.options.transform == "gaussian":
            verboseprint("Use %s as transformation" %
                         self.options.transform, verbose=self.options.verbose)
            self.phenotypes = gaussianize(self.phenotypes)
        else:
            verboseprint("Data is not transformed",
                         verbose=self.options.verbose)
        return self
