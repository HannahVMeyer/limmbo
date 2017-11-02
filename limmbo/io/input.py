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
import pdb

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
from limmbo.utils.utils import isNumericArray

########################
### functions: input ###
########################

class MissingInput(Exception):
    """Raised when no appropriate input is given"""
    pass

class FormatError(Exception):
    """Raised when no appropriate input is given"""
    pass

class DataMismatch(Exception):
    """Raised when dimensions of sample/ID names do not match dimension of 
     corresponding data"""
    pass


class DataInput(object):
    """
    Generate object containing all datasets relevant for variance decomposition
    (phenotypes, relatedness estimates) and pre-processing steps (check for
    common samples and sample order, covariates regression and phenotype 
    transformation)
    """

    def __init__(self, options=None):
        self.options = options
        self.samples = None
        self.phenotypes = None
        self.pheno_samples = None
        self.phenotype_ID = None
        self.covariates = None
        self.covariate_samples = None
        self.relatedness = None
        self.relatedness_samples = None

    def getPhenotypes(self, phenotypes=None, pheno_samples=None,
            phenotype_ID=None, verbose=False):
        """
        Add [N x P] phenotype data with [N] samples and [P] traits and their
        samples and phenotypes IDs to the DataInput object. 
        Input: 
            * In interactive/scripting: takes an np.ndarray of the phenotypes, 
              their phenotype ID and their sample ID.
                ** phenotypes: [N x P] phenotype matrix [np.array]
                ** pheno_samples: [N] sample IDs [np.array]
                ** phenotype_ID: [P] phenotype IDs [np.array]
                ** verbose: [bool] should progress messages be printed
            * In use with command-line scripts (via DataParse()): reads 
              phenotype file, either as hf5 (.h5)  or comma-separated values 
              (.csv) file; file ending must be either .h5 or .csv
                ** self.file_pheno: 
                    *** either .h5 file with group ['phenotype'] containing:
                        * ['col_header']['phenotype_ID']: [P] phenotype IDs 
                                                        [string]
                        * ['row_header']['sample_ID']: [N] sample IDs [string]
                        * ['matrix']: [N x P] phenotypes [np.array]
                    *** or [(N+1) x (P+1)] .csv file with: [N] sample IDs in 
                       the first column and [P] phenotype IDs in the first row
                ** self.option.verbose: [bool] should progress messages be 
                    printed to stdout
        Output:
            * self.phenotypes: [N x P] phenotype matrix [np.array]
            * self.pheno_samples: [N] sample IDs [np.array]
            * self.phenotype_ID: [P] phenotype IDs [np.array]
        """
        if self.options is None and phenotypes is None:
                raise MissingInput('No phenotypes specified')
        if phenotypes is not None:
            if pheno_samples is None:
                raise MissingInput('Phenotype sample names have to be',
                    'specified via pheno_samples')
            if phenotype_ID is None:
                raise MissingInput('Phenotype IDs have to be specified via',
                    ' phenotype_ID')
            if isinstance(phenotypes, np.ndarray) is False:
                raise TypeError('Phenotypes have to be of type np.ndarray', 
                    'but <{}> given'.format(type(phenotypes).__name__))
            if isinstance(pheno_samples, np.ndarray) is False:
                raise TypeError('pheno_samples has to be of type np.ndarray', 
                    'but <{}> given'.format(type(pheno_samples).__name__))
            if isinstance(phenotype_ID, np.ndarray) is False:
                raise TypeError('phenotype_ID has to be of type np.ndarray', 
                    'but <%s> given'.format(type(phenotype_ID).__name__))
	    if phenotypes.shape[0] != pheno_samples.shape[0]:
		raise DataMismatch('Number of samples in phenotypes ({}) does',
                    'not match number of sample IDs ({}) provided'.format(
                        phenotypes.shape[0], pheno_samples.shape[0]))
	    if phenotypes.shape[1] != phenotype_ID.shape[0]:
		raise DataMismatch('Number phenotypes ({}) does not match',
                    'number of phenotype IDs ({}) provided'.format(
                        phenotypes.shape[1], phenotype_ID.shape[0]))
            self.phenotypes = phenotypes
            self.pheno_samples = pheno_samples
            self.phenotype_ID = phenotype_ID
	else :
	    if re.search(".h5", self.options.file_pheno) is None \
                    and re.search(".csv", self.options.file_pheno) is None:
		raise FormatError('Supplied phenotype file is neither .h5 or', 
                    '.csv')
	    verboseprint("Reading phenotypes from %s" % self.options.file_pheno, 
			verbose=self.options.verbose)
            if re.search(".h5", self.options.file_pheno) is None:
		try:
		    self.phenotypes = pd.io.parsers.read_csv(
		    	self.options.file_pheno, index_col=0)
		except:
		    raise IOError('{} could not be opened'.format(
                        file.options.file_pheno))
		self.phenotype_ID = np.array(self.phenotypes.columns)
		self.pheno_samples = np.array(self.phenotypes.index)
		self.phenotypes = np.array(self.phenotypes)
	    else:
		try:
		    file = h5py.File(self.options.file_pheno, 'r')
		except:
		    raise IOError('{} could not be opened'.format(
                        self.options.file_pheno))
		self.phenotypes = file['phenotype']['matrix'][:]
		self.phenotype_ID = np.array(
		    file['phenotype']['col_header']['phenotype_ID'][:].astype(
			'str'))
		self.pheno_samples = np.array(
		    file['phenotype']['row_header']['sample_ID'][:].astype(
                        'str'))
		self.phenotypes = np.array(self.phenotypes)
		file.close()
        return self

    def getCovariates(self, covariates=None, covs_samples=None, verbose=False):
        """
        Add [N x K] covariate data with [N] samples and [K] covariates to 
        DataInput object. 
        Input: 
            * In interactive/scripting: takes an [N x K] np.ndarray of 
              covariates and a [N] np.ndarray of their sample IDs.
                ** covariates: [N x K] covariate matrix [np.array]
                ** covs_samples: [N] sample IDs [np.array]
                ** verbose: [bool] should progress messages be printed
            * In use with command-line scripts (via DataParse()): reads comma-
              separated values (.csv) covariates files
                ** self.file_covs: [N x (K +1)] .csv file with: [N] sample IDs 
                    in the first column
                ** self.option.verbose: [bool] should progress messages be 
                    printed to stdout
        Output:
            * self.covariates: [N x K] covariates matrix [np.array] or None if
              not set
            * self.covs_samples: [N] sample IDs [np.aray] or None of no
              covariates are specified
        """        

        if covariates is not None:
            if covs_samples is None:
                raise MissingInput('Covariate sample names have to be',
                    'specified via covs_samples')
            if isinstance(covariates, np.ndarray) is False:
                raise TypeError('Covariates have to be of type np.ndarray', 
                    'but <{}> given'.format(type(covariates).__name__))
            if isinstance(covs_samples, np.ndarray) is False:
                raise TypeError('covs_samples has to be of type np.ndarray', 
                    'but <{}> given'.format(type(covs_samples).__name__))
	    if covariates.shape[0] != covs_samples.shape[0]:
		raise DataMismatch('Number of samples in covariates ({}) does',
                    'not match number of sample IDs ({}) provided'.format(
                        covariates.shape[0], covs_samples.shape[0]))
            self.covariates = covariates
            self.covs_samples = covs_samples
	
        elif self.options.file_covariates is not None:
            if re.search(".csv", self.options.file_covariates) is None:
		raise FormatError('Supplied covariate file is not .csv')
            try:
                self.covariates = pd.io.parsers.read_csv(
                    self.options.file_covariates)
                verboseprint("Reading covariates file",
                         verbose=self.options.verbose)
            except:
                raise IOError('{} could not be opened'.format(
                    self.options.file_covariates))
            self.covs_samples = np.ravel(self.covariates.iloc[:, :1])
            self.covariates = np.array(
                self.covariates.iloc[:, 1:]).astype(float)
            # append column of 1's to adjust for mean of covariates
            self.covariates = sp.concatenate([self.covariates,
                        sp.ones((self.covariates.shape[0], 1))], 1)
            self.covariates = np.array(self.covariates)
        else:
            verboseprint("No covariates set", verbose=self.options.verbose)
            self.covariates = None
            self.covs_samples = None
        return self

    def getRelatedness(self, relatedness=None, relatedness_samples=None,
            verbose=None):
        """
        Add [N x N] pairwise relatedness estimates of [N] samples to DataInput 
        object. 
        Input: 
            * In interactive/scripting: takes an [N x N] np.ndarray of 
              rea and a [N] np.ndarray of their sample IDs.
                ** relatedness: [N x N] relatedness estimate matrix [np.array]
                ** relatedness_samples: [N] sample IDs [np.array]
                ** verbose: [bool] should progress messages be printed
            * In use with command-line scripts (via DataParse()): reads comma-
              separated values (.csv) file with relatedness estimates
                ** self.file_relatedness: [(N + 1) x N] .csv file with: [N] 
                    sample IDs in the first row
                ** self.option.verbose: [bool] should progress messages be 
                    printed to stdout
        Output:
            * self.relatedness: [N x N] relatedness matrix [np.array]
            * self.relatedness_samples: [N] sample IDs of relatedness matrix 
              [np.array]
        """
        if self.options is None and relatedness is None:
                raise MissingInput('No relatedness data specified')
        if relatedness is not None:
            if relatedness_samples is None:
                raise MissingInput('Relatedness samples names have to be',
                    'specified via relatedness_samples')
            if isinstance(relatedness, np.ndarray) is False:
                raise TypeError('Relatedness estimates have to be of type', 
                    'np.ndarray but <{}> given'.format(
                        type(relatedness).__name__))
            if isinstance(relatedness_samples, np.ndarray) is False:
                raise TypeError('relatedness_samples has to be of type',
                    ' np.ndarray but <{}> given'.format
                    (type(relatedness_samples).__name__))
	    if relatedness.shape[0] != relatedness.shape[1]:
                raise FormatError('Relatedness has to be a square matrix, but',
                    'number of rows {} is not equal to number of columns', 
                    '{}'.format(relatedness.shape[0], relatedness.shape[1]))
            if relatedness.shape[0] != relatedness_samples.shape[0]:
		raise DataMismatch('Number of samples in relatedness ({}) does',
                    'not match number of sample IDs ({}) provided'.format(
                        relatedness.shape[0], relatedness_samples.shape[0]))
            self.relatedness = relatedness
            self.relatedness_samples = relatedness_samples
        else:
            if re.search(".csv", self.options.file_relatedness) is None:
		raise FormatError('Supplied relatedness file is not .csv')
            try:
                self.relatedness = pd.io.parsers.read_csv(
                    self.options.file_relatedness)
            except:
                raise IOError('{} could not be opened'.format(
                    self.options.file_relatedness))
            verboseprint("Reading relationship matrix",
                    verbose=self.options.verbose)
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
            verboseprint('Chose subset of {} traits'.format(
                self.options.traitstring), verbose=self.options.verbose)
            search=re.compile(r'[^0-9,-]').search
            if bool(search(self.options.traitstring)):
                raise FormatError('Traitstring can only contain integers',
                        '(0-9), comma (,) and hyphen (-), but {}',
                        'provided'.format(self.options.traitstring))
            traitslist = [x.split('-')
                          for x in self.options.traitstring.split(',')]
            self.traitsarray = []
            for t in traitslist:
                if len(t) == 1:
                    self.traitsarray.append(int(t[0]) - 1)
                else:
                    [self.traitsarray.append(x) for x in range(
                        int(t[0]) - 1, int(t[1]))]
            try:
                self.phenotypes = self.phenotypes[:, self.traitsarray]
                self.phenotype_ID = self.phenotype_ID[self.traitsarray]
            except:
                raise DataMismatch('Selected trait number {} is greater', 
                        'than number of phenotypes provided {}'.format(
                        max(self.traitsarray) + 1, self.phenotypes.shape[1]))
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
        self.samples = np.intersect1d(self.pheno_samples,
                self.relatedness_samples)
        if len(self.samples) == 0:
            raise DataMismatch('No common samples between phenotypes and',
                        'relatedness estimates')
        if self.covariates is not None:
            self.samples = np.intersect1d(self.samples, self.covs_samples)
            if len(self.samples) == 0:
                raise DataMismatch('No common samples between phenotypes,',
                            'relatedness estimates and covariates')
        
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
            if np.array_equal(self.phenotypes, self.covariates):
                raise DataMismatch('Phenotype and covariate arrays are', 
                        'identical')
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
        elif self.options.transform is not None:
            raise TypeError('Possible transformation methods are: scale',
                    'gaussian or None but {} provided'.format(
                    self.options.transform))
        else:
            verboseprint("Data is not transformed",
                         verbose=self.options.verbose)
        return self
