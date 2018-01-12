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
    Generate object containing all datasets relevant for the analysis.
    For variance decomposition, at least phenotypes and relatedness estimates
    need to be specified. 
    For association testing with LMM, at least phenotype, relatedness estimates
    and genotypes need to be read.
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
	self.pcs = None
	self.pc_samples = None
	self.snps = None
	self.geno_samples = None
	self.position = None
	self.Cg = None
	self.Cn = None

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
        Read comma-separated file of [`N` x `N`] pairwise relatedness estimates 
	of [`N`] samples.

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

    def getPCs(self, file_pcs=None, nrpcs= None, verbose=True):
        r"""
        Reads a comma-separated file with [`N` x `PC`] matrix of [`PC`] 
	principal components from the genotypes of [`N`] samples.

        Arguments:
            file_pcs (string):
                [`N` x (`PC` +1)] .csv file with [`N`] sample IDs in the first
                column
	    nrpcs (integer):
		Number of PCs to use (uses first nrpcs  principal components)
            verbose (bool): should progress messages be printed to stdout

        Returns:
            None:
                updated the following attributes of the ReadData instance:

                - **self.pcs** (np.array):
                  [`N` x `PC`] principal component matrix
                - **self.pc_samples** (np.array):
                  [`N`] sample IDs

        Examples:

            .. doctest::

                >>> from pkg_resources import resource_filename
                >>> from limmbo.io.reader import ReadData
                >>> data = ReadData()
                >>> file_pcs = resource_filename('limmbo',
                ...                     'io/test/data/pcs.csv')
                >>> data.getPCs(file_pcs=file_pcs, nrpcs=10,
                ...                     verbose=False)
                >>> data.pc_samples[:3]
                array(['ID_1', 'ID_2', 'ID_3'], dtype=object)
                >>> data.pcs[:3,:3]
		array([[-0.02632738,  0.00761785,  0.01069307],
		       [-0.05791269,  0.00538956, -0.0205066 ],
		       [-0.03619099,  0.00624196,  0.02299996]])

	"""
        if  file_pcs is not None:
            verboseprint("Reading PCs", verbose=verbose)
            self.pcs = pd.io.parsers.read_csv(file_pcs,
                    header=0, sep=" ")
            self.pc_samples = np.array(self.pcs.columns)
            self.pcs = np.array(self.pcs).astype(float)
	    if nrpcs is not None:
           	verboseprint("Extracting first %s pcs" % nrpcs,
                    verbose=verbose)
            	self.pcs = self.pcs[:, :nrpcs]
        else:
            verboseprint("No pcs set", verbose=verbose)
            self.pcs=None
            self.pc_samples = None

    def getGenotypes(self, file_geno=None, verbose=False):
        r"""
        Reads genotype file, either as hf5 (.h5) or comma-separated values
        (.csv) file; file ending must be either .h5 or .csv

        Arguments:
            file_geno (string):
                path to phenotype file in hf5 or .csv format

                - **.h5 file format**: with group ['genotype'] containing:

                  - ['col_header']['chrom']: [`NrSNPs`] chromosome IDs of SNPs
                    (string)
                  - ['col_header']['pos']: [`NrSNPs`] chromosome position of 
		    SNPs (string)
                  - ['col_header']['rs']: [`NrSNPs`] rsIDs/IDs of SNPs (string)
                  - ['row_header']['sample_ID']: [`N`] sample IDs (string)
                  - ['matrix']: [`N` x `NrSNP`] genotypes (array-like)

                - **.csv format**:

                  - [(`NrSNP` + 1) x (`N`+1)] .csv file with: [`N`] sample IDs 
                    in the first row and [`NrSNP`] genotype IDs in the first 
		    column
		  - sample IDs should be of type: chrom-pos-rsID for instance
		    22-50714616-rs6010226
		  
            verbose (bool):
                should progress messages be printed to stdout

        Returns:
            None:
                updated the following attributes of the ReadData instance:

                - **self.snps** (np.array):
                  [`N` x `NrSNPs`] genotype matrix
                - **self.geno_samples** (np.array):
                  [`N`] sample IDs
                - **self.position** (pd.dataframe):
                  [`NrSNPs` x 2] dataframe with columns 'chrom' and 'pos', and
		  rsIDs as index

        Examples:

            .. doctest::

                >>> from pkg_resources import resource_filename
                >>> from limmbo.io.reader import ReadData
                >>> data = ReadData()
                >>> file_geno = resource_filename('limmbo',
                ...                                'io/test/data/genotypes.csv')
                >>> data.getGenotypes(file_geno=file_geno,
                ...                   verbose=False)
                >>> data.geno_samples[:5]
                array(['ID_1', 'ID_2', 'ID_3', 'ID_4', 'ID_5'], dtype=object)
		>>> data.snps.shape
		(1000, 20)
                >>> data.snps[:5,:5]
		array([[ 0.,  1.,  1.,  0.,  1.],
		       [ 0.,  1.,  0.,  1.,  1.],
		       [ 0.,  0.,  0.,  0.,  0.],
		       [ 0.,  1.,  0.,  0.,  1.],
		       [ 0.,  0.,  1.,  0.,  0.]])
		>>> data.position[:5]
			    chrom       pos
		rs111647458    15  49385160
		rs67918533     15  92151569
		rs12903533     15  98887790
		rs34937778     19  18495997
		rs150270350    19  47869060

        """

	if re.search(".h5", file_geno) is None \
		and re.search(".csv", file_geno) is None:
	    raise FormatError('Supplied genotype file is neither .h5 or',
		'.csv')
	verboseprint("Reading genotypes from %s" % file_geno,
		    verbose=verbose)
	if re.search(".h5", file_geno) is None:
	    try:
            	genotypes = pd.io.parsers.read_csv(file_geno,
                    index_col=0, header=0)
	    except:
		raise IOError('{} could not be opened'.format(
		    file_geno))

            snp_info = np.array(genotypes.index)

            position = []
            snp_ID = []
            for id in range(snp_info.shape[0]):
                split = np.array(snp_info[id].split('-'))
                snp_ID.append(split[2])
                position.append(split[[0,1]])

            self.geno_samples = np.array(genotypes.columns)
            self.snps = np.array(genotypes).astype(float).T
            self.position = pd.DataFrame(np.array(position),
                    columns=['chrom', 'pos'], index=snp_ID)
        else:
            # read genotype information: chromosome-wise hf5 files
            geno_reader  = gr.genotype_reader_h5py(file_geno)
            verboseprint("Extracting genotypes from hf5 file",
                    verbose=self.verbose)
            self.geno_samples = geno_reader.sample_ID
            self.snps = geno_reader.getGenotypes().astype(float)
            self.position = geno_reader.getPos()



    def getVarianceComponents(self, file_Cg=None, file_Cn=None, verbose=True):
        r"""
        Reads a comma-separated files with [`P` x `P`] matrices of [`P`] trait
	covariance estimates. 

        Arguments:
            file_Cg (string):
                [`P` x `P`] .csv file with [`P`] trait covariance estimates of
                the genetic component
            file_Cn (string):
                [`P` x `P`] .csv file with [`P`] trait covariance estimates of
                the non-genetic (noise) component
            verbose (bool): should progress messages be printed to stdout

        Returns:
            None:
                updated the following attributes of the ReadData instance:

                - **self.Cg** (np.array):
                  [`P` x `P`] matrix with trait covariance of the genetic 
		  component
                - **self.Cn** (np.array):
                  [`P` x `P`] matrix with trait covariance of the non-genetic 
		  (noise) component

        Examples:

            .. doctest::

                >>> from pkg_resources import resource_filename
                >>> from limmbo.io.reader import ReadData
                >>> data = ReadData()
                >>> file_Cg = resource_filename('limmbo',
                ...                     'io/test/data/Cg.csv')
                >>> file_Cn = resource_filename('limmbo',
                ...                     'io/test/data/Cn.csv')
                >>> data.getVarianceComponents(file_Cg=file_Cg, 
		...			       file_Cn=file_Cn,
                ...                            verbose=False)
		>>> data.Cg.shape
		(10, 10)
		>>> data.Cn.shape
		(10, 10)
                >>> data.Cg[:3,:3]
		array([[ 0.04265732, -0.02540865,  0.01784288],
		       [-0.02540865,  0.03610362, -0.02802982],
		       [ 0.01784288, -0.02802982,  0.04448125]])
                >>> data.Cn[:3,:3]
		array([[ 0.96301131, -0.86185094, -0.35197147],
		       [-0.86185094,  0.9647436 ,  0.37166803],
		       [-0.35197147,  0.37166803,  0.96285129]])
	"""
	
	if file_Cg is None and file_Cn is None:
	    verboseprint(("No variance components supplied, run VD/limmbo",
	    "before lmm test"), verbose=verbose)
	elif file_Cg is None or file_Cn is None:
	    raisIOError('Both variant components need to be supplied:',
	    		'Cg is %s and Cn is %s') % (file_Cg, file_Cn)
	else:
	    try:
	    	self.Cg = np.array(pd.io.parsers.read_csv(file_Cg, header=None))
	    except:
		raise IOError('{} could not be opened'.format(file_Cg))
	    try:
	    	self.Cn = np.array(pd.io.parsers.read_csv(file_Cn, header=None))
	    except:
		raise IOError('{} could not be opened'.format(file_Cn))





