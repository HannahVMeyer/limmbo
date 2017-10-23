###############
### modules ###
###############

import h5py

import sys
sys.path.append('./../../')


import scipy as sp
import pandas as pd
import numpy as np
import functools as ft
import re

# import LIMIX tools
import limix as limix
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
    Generate object containing all datasets relevant for variance decomposition
    (phenotypes, relatedness estimates) and pre-processing steps (check for
    common samples and sample order, covariates regression and phenotype 
    transformation)
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
        self.covariates = None
        self.covariate_samples = None
        self.relatedness = None
        self.relatedness_samples = None

    def getPhenotypes(self):
        """
        Reading phenotype file, either as hf5 (.h5)  or comma-separated
        values (.csv) file
        Input: 
            * self.file_pheno: 
                * either .h5f file with group ['phenotype'] containing:
                    * ['col_header']['phenotype_ID']: [P] phenotype IDs [string]
                    * ['row_header']['sample_ID']: [N] sample IDs [string]
                    * ['matrix']: [N x P] phenotypes [np.array]
                * or [(N+1) x (P+1)] .csv file with: [N] sample IDs in the 
		  first column and [P] phenotype IDs in the first row
            * self.option.verbose: [bool] should progress messages be printed
              to stdout
        Output:
            * self.phenotypes: [N x P] phenotype matrix [np.array]
            * self.pheno_samples: [N] sample IDs [np.array]
            * self.phenotype_ID: [P] phenotype IDs [np.array]
        """
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

    def getCovariates(self):
        """
        Reading comma-separated values (.csv) covariates file with [N]
	samples and [K] covariates
        Input:
            * self.file_covariates: [N x (K + 1)] .csv file with: [N] sample 
	     IDs in the first column 
            * self.option.verbose: [bool] should progress messages be printed
              to stdout
        Output:
            * self.covariates: [N x K] covariates matrix [np.array] or None if
              not set
            * self.covs_samples: [N] sample IDs [np.aray] or None of no
              covariates are specified
        """        

	if self.options.file_covariates != None:
            verboseprint("Reading covariates file",
                         verbose=self.options.verbose)
            self.covariates = pd.io.parsers.read_csv(
                self.options.file_covariates)
            self.covs_samples = np.ravel(self.covariates.iloc[:, :1])
            self.covariates = np.array(
                self.covariates.iloc[:, 1:]).astype(float)
            # append column of 1's to adjust for mean of covariates
            if not self.options.regress:
                self.covariates = sp.concatenate(
                    [self.covariates,
                        sp.ones((self.covariates.shape[0], 1))], 1)
                self.covariates = np.array(self.covariates)
        else:
            verboseprint("No covariates set", verbose=self.options.verbose)
            self.covariates = None
            self.covs_samples = None
        return self

    def getRelatedness(self):
        """
        Reading comma-separated values (.csv) file with relatedness estimates 
        for [N] samples 
        Input:
            * self.file_relatedness: [(N + 1) x N] .csv file with: [N] sample
              IDs in the first row
            * self.options.verbose: [bool] should progress messages be printed
              to stdout
        Output:
            * self.relatedness: [N x N] relatedness matrix [np.array]
            * self.relatedness_samples: [N] sample IDs of relatedness matrix 
              [np.array]
        """
        verboseprint("Reading relationship matrix",
                     verbose=self.options.verbose)
        self.relatedness = pd.io.parsers.read_csv(
            self.options.file_relatedness)
        self.relatedness_samples = np.array(self.relatedness.columns)
        self.relatedness = np.array(self.relatedness).astype(float)
        return self

    def subsetTraits(self):
        """
        Limit analysis to specific subset of traits
        Input:
            * self.phenotypes: [N x P] phenotype matrix [np.array]
            * self.phenotype_ID: [P] phenotype IDs [np.array]
            * self.options.traitstring: comma-separated trait numbers (for 
              single traits) or hyphen-separated trait numbers 
              (for trait ranges) or combination of both [string] for trait
              selection (1-based) 
            * self.options.verbose: [bool] should progress messages be printed
              to stdout
        Output:
            * self.traitsarray: [list] of [t] trait numbers [int] to choose for 
              analysis
            * self.phenotypes: reduced set of [N x t] phenotypes [np.array]
            * self.phenotype.ID: reduced set of [t] phenotype IDs [np.array]
        """
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
        return self

    def commonSamples(self):
        """
        Get [M] common samples out of phenotype, relatedness and optional 
        covariates with [N] samples (if all the same [M] = [N])
        and ensure that samples are in same order
        Input:
            * self.phenotypes: [N x P] phenotype matrix [np.array]
            * self.pheno_samples: [N] sample IDs [np.array]
            * self.relatedness: [N x N] relatedness matrix [np.array]
            * self.relatedness_samples: [N] sample IDs of relatedness matrix 
              [np.array]
            * self.covariates: [N x K] covariates matrix [np.array]
            * self.covs_samples: [N] sample IDs [np.aray]
        Output:
            * self.phenotypes: [M x P] phenotype matrix [np.array]
            * self.pheno_samples: [M] sample IDs [np.array]
            * self.relatedness: [M x M] relatedness matrix [np.array]
            * self.relatedness_samples: [M] sample IDs of relatedness matrix 
              [np.array]
            * self.covariates: [M x K] covariates matrix [np.array]
            * self.covs_samples: [M] sample IDs [np.aray]
        """
        if self.covariates is not None:
            self.samples = ft.reduce(np.intersect, (self.pheno_samples,
                    self.relatedness_samples, self.covs_samples))
        else:
            self.samples = np.intersect(self.pheno_samples,
                    self.relatedness_samples)
        
        subset_pheno_samples = np.in1d(
            self.pheno_samples, self.samples)
        self.pheno_samples = self.pheno_samples[
            subset_pheno_samples]
        self.phenotypes = self.phenotypes[subset_pheno_samples, :]
        (self.phenotypes, self.pheno_samples, samples_before,
         samples_after) = match(
            self.samples, self.pheno_samples, self.phenotypes,
            squarematrix=True)

        subset_relatedness_samples = np.in1d(
            self.relatedness_samples, self.samples)
        self.relatedness_samples = self.relatedness_samples[
            subset_relatedness_samples]
        self.relatedness = self.relatedness[subset_relatedness_samples,:]\
                [:, subset_relatedness_samples]
        (self.relatedness, self.relatedness_samples, samples_before,
         samples_after) = match(
            self.samples, self.relatedness, self.relatedness_samples,
            squarematrix=True)

        if self.covariates is not None:
            subset_covs_samples = np.in1d(self.covs_samples, self.samples)
            self.covs_samples = self.covs_samples[subset_covs_samples]
            self.covariates = self.covariates[subset_covs_samples, :]
            self.covariates, self.covs_samples, samples_before,
            samples_after = match(
                self.samples, self.covariates, self.covs_samples,
                squarematrix=False)
        return self

    def regress_and_transform(self):
        """
        Regress out covariates (optional) and transform phenotypes (optional)
        Input:
            * self.options.regress: [bool], if True, covariates are explanatory
              variables in linear model with phenotypes as response
            * self.phenotypes: [N x P] phenotype matrix [np.array]
            * self.covariates: [N x K] covariates matrix [np.array]
            * self.options.verbose: [bool] should progress messages be printed
              to stdout
        Output: 
            * self.options.phenotypes: [N x P] phenotype matrix of residuals of
              linear model [np.array]
            * self.covariates: None 
        """

        if self.options.regress:
            type = "covariates"
            verboseprint("Regress out %s" %
                         type,  verbose=self.options.verbose)
            self.phenotypes = regressOut(self.phenotypes, self.covariates)
            self.covariates = None
            self.transform()
        return self

    def transform(self):
        """
        Transform phenotypes
        Input:
            * self.options.tranform: type of transformation for phenotype data:
                ** scale: mean center, divide by sd
                ** gaussian: inverse normalisation
                ** None: no transformation
            * self.phenotypes: [N x P] phenotype matrix [np.array]
            * self.options.verbose: [bool] should progress messages be printed
              to stdout
        Output:
            * self.phenotypes: [N x P] (transformed) phenotype matrix 
              [np.array] 
        """
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
