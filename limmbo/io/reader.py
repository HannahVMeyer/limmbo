###############
### modules ###
###############

import h5py

import scipy as sp
import pandas as pd
import numpy as np
import re

# other requirements
from math import sqrt
from limmbo.utils.utils import verboseprint

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


class ReadData(object):
    r"""
    Generate object containing all datasets relevant for variance decomposition
    (phenotypes, relatedness estimates) and pre-processing steps (check for
    common samples and sample order, covariates regression and phenotype
    transformation)
    """

    def __init__(self):
        self.samples = None
        self.phenotypes = None
        self.pheno_samples = None
        self.phenotype_ID = None
        self.covariates = None
        self.covs_samples = None
        self.relatedness = None
        self.relatedness_samples = None

    def getPhenotypes(self, file_pheno=None,
                     verbose=True):
        r"""
        Reads phenotype file, either as hf5 (.h5)  or comma-separated values 
        (.csv) file; file ending must be either .h5 or .csv

        Arguments:
            file_pheno (string): 
                path to phenotype file in hf5 or .csv format

                - **.h5 file format**: with group ['phenotype'] containing:
                  
                  - ['col_header']['phenotype_ID']: [`P`] phenotype IDs
                    (string) 
                  - ['row_header']['sample_ID']: [`N`] sample IDs (string)
                  - ['matrix']: [`N` x `P`] phenotypes (array-like)
                
                - **.csv format**:
                  [(`N`+1) x (`P`+1)] .csv file with: [`N`] sample IDs in 
                  the first column and [`P`] phenotype IDs in the first row
            verbose (bool):
                should progress messages be printed to stdout

        Returns:
            None:
                updated the following attributes of the ReadData instance:

                - **self.phenotypes** (np.array): 
                  [`N` x `P`] phenotype matrix
                - **self.pheno_samples** (np.array): 
                  [`N`] sample IDs
                - **self.phenotype_ID** (np.array): 
                  [`P`] phenotype IDs

        Examples:
        
            .. doctest::

                >>> from pkg_resources import resource_filename
                >>> from limmbo.io.reader import ReadData
                >>> data = ReadData()
                >>> file_pheno = resource_filename('limmbo', 
                ...                                'io/test/data/pheno.csv')
                >>> data.getPhenotypes(file_pheno=file_pheno,
                ...                   verbose=False)
                >>> data.pheno_samples[:3]
                array(['ID_1', 'ID_2', 'ID_3'], dtype=object)
                >>> data.phenotype_ID[:3]
                array(['trait_1', 'trait_2', 'trait_3'], dtype=object)
                >>> data.phenotypes[:3,:3]
                array([[-0.58292258, -0.64127589,  1.02820392],
                       [ 1.40385214, -1.33475008, -0.85868719],
                       [-0.14518214,  0.53119702, -0.98530978]])
        """

        if file_pheno is None:
            raise MissingInput('No phenotype file specified')
	else :
	    if re.search(".h5", file_pheno) is None \
                    and re.search(".csv", file_pheno) is None:
		raise FormatError('Supplied phenotype file is neither .h5 or', 
                    '.csv')
	    verboseprint("Reading phenotypes from %s" % file_pheno, 
			verbose=verbose)
            if re.search(".h5", file_pheno) is None:
		try:
		    self.phenotypes = pd.io.parsers.read_csv(
		    	file_pheno, index_col=0)
		except:
		    raise IOError('{} could not be opened'.format(
                        file_pheno))
		self.phenotype_ID = np.array(self.phenotypes.columns)
		self.pheno_samples = np.array(self.phenotypes.index)
		self.phenotypes = np.array(self.phenotypes)
	    else:
		try:
		    file = h5py.File(file_pheno, 'r')
		except:
		    raise IOError('{} could not be opened'.format(
                        file_pheno))
		self.phenotypes = file['phenotype']['matrix'][:]
		self.phenotype_ID = np.array(
		    file['phenotype']['col_header']['phenotype_ID'][:].astype(
			'str'))
		self.pheno_samples = np.array(
		    file['phenotype']['row_header']['sample_ID'][:].astype(
                        'str'))
		self.phenotypes = np.array(self.phenotypes)
		file.close()

    def getCovariates(self, file_covariates=None, verbose=True):
        r"""
        Reads a comma-separated [`N` x `K`] covariate matrix with [`N`] samples
        and [`K`] covariates

        Arguments: 
            file_covariates (string): 
                [`N` x (`K` +1)] .csv file with [`N`] sample IDs in the first 
                column
            verbose (bool): should progress messages be printed to stdout

        Returns:
            None:
                updated the following attributes of the ReadData instance:

                - **self.covariates** (np.array): 
                  [`N` x `K`] covariates matrix 
                - **self.covs_samples** (np.array):
                  [`N`] sample IDs
        
        Examples:
        
            .. doctest::

                >>> from pkg_resources import resource_filename
                >>> from limmbo.io.reader import ReadData
                >>> data = ReadData()
                >>> file_covs = resource_filename('limmbo', 
                ...                               'io/test/data/covs.csv')
                >>> data.getCovariates(file_covariates=file_covs,
                ...                   verbose=False)
                >>> data.covs_samples[:3]
                array(['ID_1', 'ID_2', 'ID_3'], dtype=object)
                >>> data.covariates[:3,:3]
                array([[-1.05808516, -0.89731694,  0.18733211],
                       [ 0.28205298,  0.57994795, -0.41383724],
                       [-1.55179427, -1.70411737, -0.448364  ]])
        """

	
        if file_covariates is not None:
            if re.search(".csv", file_covariates) is None:
		raise FormatError('Supplied covariate file is not .csv')
            try:
                self.covariates = pd.io.parsers.read_csv(file_covariates)
                verboseprint("Reading covariates file", verbose=verbose)
            except:
                raise IOError('{} could not be opened'.format(
                    file_covariates))
            self.covs_samples = np.ravel(self.covariates.iloc[:, :1])
            self.covariates = np.array(
                self.covariates.iloc[:, 1:]).astype(float)
            # append column of 1's to adjust for mean of covariates
            self.covariates = sp.concatenate([self.covariates,
                        sp.ones((self.covariates.shape[0], 1))], 1)
            self.covariates = np.array(self.covariates)
        else:
            verboseprint("No covariates set", verbose=verbose)
            self.covariates = None
            self.covs_samples = None

    def getRelatedness(self, file_relatedness, verbose=True):
        """
        Read comma-separated [`N` x `N`] pairwise relatedness estimates of 
        [`N`] samples.

        Arguments: 
            file_relatedness (string): 
                [(`N` + `1`) x N] .csv file with: [`N`] sample IDs in the first 
                row
            verbose (bool): 
                should progress messages be printed to stdout

        Returns:
            None:
                updated the following attributes of the ReadData instance:

                - **self.relatedness** (np.array): 
                  [`N` x `N`] relatedness matrix 
                - **self.relatedness_samples** (np.array): 
                  [`N`] sample IDs of relatedness matrix
        Examples:
        
            .. doctest::

                >>> from pkg_resources import resource_filename
                >>> from limmbo.io.reader import ReadData
                >>> data = ReadData()
                >>> file_relatedness = resource_filename('limmbo', 
                ...                     'io/test/data/relatedness.csv')
                >>> data.getRelatedness(file_relatedness=file_relatedness,
                ...                     verbose=False)
                >>> data.relatedness_samples[:3]
                array(['ID_1', 'ID_2', 'ID_3'], dtype=object)
                >>> data.relatedness[:3,:3]
                array([[  1.00882922e+00,   2.00758504e-04,   4.30499103e-03],
                       [  2.00758504e-04,   9.98844885e-01,   4.86487318e-03],
                       [  4.30499103e-03,   4.86487318e-03,   9.85687665e-01]])
        """
        if file_relatedness is None:
                raise MissingInput('No relatedness data specified')
        if re.search(".csv", file_relatedness) is None:
            raise FormatError('Supplied relatedness file is not .csv')
        try:
            self.relatedness = pd.io.parsers.read_csv(file_relatedness)
        except:
            raise IOError('{} could not be opened'.format(
            file_relatedness))
        verboseprint("Reading relationship matrix",
                verbose=verbose)
        self.relatedness_samples = np.array(self.relatedness.columns)
        self.relatedness = np.array(self.relatedness).astype(float)

