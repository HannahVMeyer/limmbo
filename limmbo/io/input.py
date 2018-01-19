r"""Title
---------

Document

.. autoclass::DataInput
    :members:
"""

###############
### modules ###
###############

import sys

# import LIMIX tools
from limix.util.preprocess import regressOut

# import LiMMBo tools
from limmbo.utils.utils import verboseprint
from limmbo.utils.utils import match
from limmbo.utils.utils import scale

# other requirements
import pandas as pd
import numpy as np
import re
from scipy_sugar.stats import quantile_gaussianize

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


class InputData(object):
    """
    Generate object containing all datasets relevant for variance decomposition
    (phenotypes, relatedness estimates) and pre-processing steps (check for
    common samples and sample order, covariates regression and phenotype
    transformation)
    """

    def __init__(self, verbose=True):
        self.samples = None
        self.phenotypes = None
        self.pheno_samples = None
        self.phenotype_ID = None
        self.covariates = None
        self.covariate_samples = None
        self.relatedness = None
        self.relatedness_samples = None
        self.pcs = None
        self.pc_samples = None
        self.snps = None
        self.genotypes = None
        self.geno_samples = None
        self.position = None
        self.Cg = None
        self.Cn = None

    def addPhenotypes(self,
                      phenotypes,
                      pheno_samples,
                      phenotype_ID,
                      verbose=True):
        r"""
        Add phenotypes, their phenotype ID and their sample IDs to
        InputData instance

        Arguments:
            phenotypes (array-like):
                [`N x `P`] phenotype matrix of `N` individuals and `P`
                phenotypes
            pheno_samples (array-like):
                [`N`] sample ID
            phenotype_ID (array-like):
                [`P`] phenotype IDs
            verbose: bool
                should progress messages be printed to stdout

        Returns:
            None:
                updated the following attributes of the InputData instance:

                - **self.phenotypes** (np.array):
                  [`N` x `P`] phenotype matrix
                - **self.pheno_samples** (np.array):
                  [`N`] sample IDs
                - **self.phenotype_ID** (np.array):
                  [`P`] phenotype IDs


        Examples:

            .. doctest::

                >>> from limmbo.io import input
                >>> import numpy as np
                >>> phenotypes = np.array(((1,2),(7,1),(3,4)))
                >>> pheno_samples = np.array(('S1','S2', 'S3'))
                >>> phenotype_ID = np.array(('ID2','ID2'))
                >>> indata = input.InputData()
                >>> indata.addPhenotypes(phenotypes = phenotypes,
                ...                      pheno_samples = pheno_samples,
                ...                      phenotype_ID = phenotype_ID)
                >>> print indata.phenotypes.shape
                (3, 2)
                >>> print indata.pheno_samples.shape
                (3,)
                >>> print indata.phenotype_ID.shape
                (2,)
        """
        if phenotypes is None:
            raise MissingInput('No phenotypes specified')
        if pheno_samples is None:
            raise MissingInput('Phenotype sample names have to be',
                               'specified via pheno_samples')
        if phenotype_ID is None:
            raise MissingInput('Phenotype IDs have to be specified via',
                               ' phenotype_ID')

        self.phenotypes = np.array(phenotypes)
        self.pheno_samples = np.array(pheno_samples)
        self.phenotype_ID = np.array(phenotype_ID)

        if self.phenotypes.shape[0] != self.pheno_samples.shape[0]:
            raise DataMismatch(
                'Number of samples in phenotypes ({}) does',
                'not match number of sample IDs ({}) provided'.format(
                    self.phenotypes.shape[0], self.pheno_samples.shape[0]))
        if self.phenotypes.shape[1] != self.phenotype_ID.shape[0]:
            raise DataMismatch('Number phenotypes ({}) does not match',
                               'number of phenotype IDs ({}) provided'.format(
                                   self.phenotypes.shape[1],
                                   self.phenotype_ID.shape[0]))

    def addCovariates(self, covariates=None, covs_samples=None, verbose=True):
        r"""
        Add [`N` x `K`] covariate data with [`N`] samples and [`K`] covariates
        to InputData instance.

        Arguments:
            covariates (array-like):
                [`N x `K`] covariate matrix of `N` individuals and `K`
                covariates
            covs_samples (array-like):
                [`N`] sample ID
            verbose (bool):
                should progress messages be printed to stdout

	Returns:
            None:
                updated the following attributes of the InputData instance:

                - **self.covariates** (np.array):
                  [`N` x `K`] covariates matrix
                - **self.covs_samples** (np.array):
                  [`N`] sample IDs

        Examples:

            .. doctest::

                >>> from limmbo.io import input
                >>> import numpy as np
                >>> covariates = np.array(((1,2,4),(1,1,6),(0,4,8)))
                >>> covs_samples = np.array(('S1','S2', 'S3'))
                >>> indata = input.InputData()
                >>> indata.addCovariates(covariates = covariates,
                ...                      covs_samples = covs_samples)
                >>> print indata.covariates.shape
                (3, 3)
                >>> print indata.covs_samples.shape
                (3,)

        """

        if covariates is not None:
            if covs_samples is None:
                raise MissingInput('Covariate sample names have to be',
                                   'specified via covs_samples')
            if np.array(covariates).shape[0] != np.array(
                    covs_samples).shape[0]:
                raise DataMismatch(
                    'Number of samples in covariates ({}) does',
                    'not match number of sample IDs ({}) provided'.format(
                        np.array(covariates).shape[0],
                        np.array(covs_samples).shape[0]))
            self.covariates = np.array(covariates)
            self.covs_samples = np.array(covs_samples)
        else:
            verboseprint("No covariates set", verbose=verbose)

    def addRelatedness(self,
                       relatedness,
                       relatedness_samples=None,
                       verbose=True):
        """
        Add [`N` x `N`] pairwise relatedness estimates of [`N`] samples to the
	InputData instance

        Arguments:
            relatedness (array-like):
                [`N x `N`] relatedness matrix of `N` individuals
            relatedness_samples (array-like):
                [`N`] sample IDs
            verbose (bool):
                should progress messages be printed to stdout

	Returns:
            None:
                updated the following attributes of the InputData instance:

                - **self.relatedness** (np.array):
                  [`N` x `N`] relatedness matrix
                - **self.relatedness_samples** (np.array):
                  [`N`] sample IDs

        Examples:

            .. doctest::

                >>> from limmbo.io import input
		>>> import numpy
		>>> from numpy.random import RandomState
		>>> from numpy.linalg import cholesky as chol
		>>> random = RandomState(5)
		>>> N = 100
		>>> SNP = 1000
		>>> X = (random.rand(N, SNP) < 0.3).astype(float)
                >>> relatedness = numpy.dot(X, X.T)/float(SNP)
                >>> relatedness_samples = numpy.array(
                ...     ['S{}'.format(x+1) for x in range(N)])
                >>> indata = input.InputData()
                >>> indata.addRelatedness(relatedness = relatedness,
                ...                  relatedness_samples = relatedness_samples)
                >>> print indata.relatedness.shape
                (100, 100)
                >>> print indata.relatedness_samples.shape
                (100,)

        """
        if relatedness is None:
            raise MissingInput('No relatedness data specified')
        if relatedness_samples is None:
            raise MissingInput('Relatedness samples names have to be',
                               'specified via relatedness_samples')
        self.relatedness = np.array(relatedness)
        self.relatedness_samples = np.array(relatedness_samples)

        if self.relatedness.shape[0] != self.relatedness.shape[1]:
            raise FormatError(
                'Relatedness has to be a square matrix, but',
                'number of rows {} is not equal to number of columns',
                '{}'.format(self.relatedness.shape[0],
                            self.relatedness.shape[1]))
        if self.relatedness.shape[0] != self.relatedness_samples.shape[0]:
            raise DataMismatch(
                'Number of samples in relatedness ({}) does',
                'not match number of sample IDs ({}) provided'.format(
                    self.relatedness.shape[0],
                    self.relatedness_samples.shape[0]))

    def addGenotypes(self,
                     genotypes=None,
                     geno_samples=None,
                     genotypes_info=None,
                     verbose=True):
        """
        Add [`N` x `NrSNP`] genotype array of [`N`] samples and [`NrSNP`]
	genotypes, [`N`] array of sample IDs and [`NrSNP` x 2] dataframe of
	genotype description to InputData instance.

        Arguments:
            genotypes (array-like):
                [`N` x `NrSNP`] genotype array of [`N`] samples and [`NrSNP`]
        	genotypes
            geno_samples (array-like):
                [`N`] vector of `N` sample IDs
            genotypes_info (dataframe):
                  [`NrSNPs` x 2] dataframe with columns 'chrom' and 'pos', and
                  rsIDs as index
            verbose (bool):
                should progress messages be printed to stdout

        Returns:
            None:
                updated the following attributes of the InputData instance:

		- **self.genotypes** (np.array):
                  [`N` x `NrSNPs`] genotype matrix
                - **self.geno_samples** (np.array):
                  [`N`] sample IDs
                - **self.genotypes_info** (pd.dataframe):
                  [`NrSNPs` x 2] dataframe with columns 'chrom' and 'pos', and
                  rsIDs as index

        Examples:

            .. doctest::

		>>> from pkg_resources import resource_filename
                >>> from limmbo.io import reader
                >>> from limmbo.io import input
                >>> data = reader.ReadData()
                >>> file_geno = resource_filename('limmbo',
                ...                                'io/test/data/genotypes.csv')
                >>> data.getGenotypes(file_geno=file_geno,
                ...                   verbose=False)
                >>> indata = input.InputData()
                >>> indata.addGenotypes(genotypes=data.genotypes,
		...			genotypes_info=data.genotypes_info,
		...			geno_samples=data.geno_samples)
                >>> indata.geno_samples[:5]
                array(['ID_1', 'ID_2', 'ID_3', 'ID_4', 'ID_5'], dtype=object)
                >>> indata.genotypes.shape
                (1000, 20)
                >>> indata.genotypes[:5,:5]
                array([[ 0.,  1.,  1.,  0.,  1.],
                       [ 0.,  1.,  0.,  1.,  1.],
                       [ 0.,  0.,  0.,  0.,  0.],
                       [ 0.,  1.,  0.,  0.,  1.],
                       [ 0.,  0.,  1.,  0.,  0.]])
                >>> indata.genotypes_info[:5]
                            chrom       pos
                rs111647458    15  49385160
                rs67918533     15  92151569
                rs12903533     15  98887790
                rs34937778     19  18495997
                rs150270350    19  47869060
	"""
        if geno_samples is None:
            raise MissingInput('Genotype sample names have to be',
                               'specified via geno_samples')
        if genotypes_info is None:
            raise MissingInput('Genotype info has to be specified via',
                               ' genotypes_info')
        self.genotypes = np.array(genotypes)
        self.genotypes_info = pd.DataFrame(genotypes_info)
        self.geno_samples = np.array(geno_samples)

        if self.genotypes.shape[0] != self.geno_samples.shape[0]:
            raise DataMismatch(
                'Number of samples in genotypes ({}) does',
                'not match number of sample IDs ({}) provided'.format(
                    self.genotypes.shape[0], self.geno_samples.shape[0]))

        if self.genotypes.shape[1] != self.genotypes_info.shape[0]:
            raise DataMismatch(
                'Number of genotypes in genotypes ({}) does',
                'not match number of genotypes in genotypes_info ({})'.format(
                    self.genotypes.shape[1], self.genotypes_info.shape[0]))

    def addVarianceComponents(self, Cg=None, Cn=None, verbose=True):
        """
        Add [`P` x `P`] matrices of [`P`] trait covariance estimates
	of the genetic trait variance component (Cg) and the non-genetic (noise)
	variance component (Cn) to InputData instance

        Arguments:
            Cg (array-like):
                [`P x `P`] matrix of `P` trait covariance estimates of the
		genetic trait covaraince component
            Cn (array-like):
                [`P x `P`] matrix of `P` trait covariance estimates of the
		non-genetic (noise) trait covaraince component
            verbose (bool):
                should progress messages be printed to stdout

        Returns:
            None:
                updated the following attributes of the InputData instance:

                - **self.Cg** (np.array):
                  [`P x `P`] matrix of `P` trait covariance estimates of the
		  genetic trait covaraince component
                - **self.Cn** (np.array):
                  [`P x `P`] matrix of `P` trait covariance estimates of the
		  non-genetic trait covaraince component

        Examples:

            .. doctest::

                >>> from pkg_resources import resource_filename
                >>> from limmbo.io import reader
                >>> from limmbo.io import input
                >>> import numpy as np
                >>> from numpy.random import RandomState
                >>> from numpy.linalg import cholesky as chol
                >>> data = reader.ReadData()
                >>> file_pheno = resource_filename('limmbo',
                ...                     'io/test/data/pheno.csv')
                >>> data.getPhenotypes(file_pheno=file_pheno,
                ...                            verbose=False)
                >>> file_Cg = resource_filename('limmbo',
                ...                     'io/test/data/Cg.csv')
                >>> file_Cn = resource_filename('limmbo',
                ...                     'io/test/data/Cn.csv')
                >>> data.getVarianceComponents(file_Cg=file_Cg,
                ...                            file_Cn=file_Cn,
                ...                            verbose=False)
                >>> indata = input.InputData()
                >>> indata.addPhenotypes(phenotypes = data.phenotypes,
	  	...			 pheno_samples = data.pheno_samples,
		...			 phenotype_ID = data.phenotype_ID)
                >>> indata.addVarianceComponents(Cg = data.Cg, Cn=data.Cn)
                >>> print indata.Cg.shape
                (10, 10)
                >>> print indata.Cg.shape
                (10, 10)
	"""
        if self.phenotypes is None:
            raise FormatError('Phenotypes have to be added before Cg/Cn',
                              'can be added')
        self.Cg = np.array(Cg)
        self.Cn = np.array(Cn)

        if self.Cg.shape[0] != self.Cg.shape[1]:
            raise FormatError(
                'Cg has to be a square matrix, but',
                'number of rows {} is not equal to number of columns',
                '{}'.format(self.Cg.shape[0], self.Cg.shape[1]))
        if self.Cg.shape[0] != self.phenotypes.shape[1]:
            raise DataMismatch(
                'Number of traits in Cg ({}) does',
                'not match number of traits ({}) in phenotypes'.format(
                    self.Cg.shape[0], self.phenotypes.shape[1]))

        if self.Cn.shape[0] != self.Cn.shape[1]:
            raise FormatError(
                'Cn has to be a square matrix, but',
                'number of rows {} is not equal to number of columns',
                '{}'.format(self.Cn.shape[0], self.Cn.shape[1]))
        if self.Cn.shape[0] != self.phenotypes.shape[1]:
            raise DataMismatch(
                'Number of traits in Cn ({}) does',
                'not match number of traits ({}) in phenotypes'.format(
                    self.Cn.shape[0], self.phenotypes.shape[1]))

    def addPCs(self, pcs=None, pc_samples=None, verbose=True):
        r"""
	Add [`N` x `PC`] matrix of [`PC`] principal components from the
	genotypes of [`N`] samples to InputData instance.

        Arguments:
            pcs (array-like):
                [`N x `PCs`] principal component matrix of `N` individuals and
		`PCs` principal components
            pc_samples (array-like):
                [`N`] sample IDs
            verbose (bool):
                should progress messages be printed to stdout

	Returns:
            None:
                updated the following attributes of the InputData instance:

                - **self.pcs** (np.array):
                  [`N` x `PCs`] principal component matrix
                - **self.pc_samples** (np.array):
                  [`N`] sample IDs

        Examples:

            .. doctest::

                >>> from pkg_resources import resource_filename
                >>> from limmbo.io import reader
                >>> from limmbo.io import input
                >>> data = reader.ReadData()
                >>> file_pcs = resource_filename('limmbo',
                ...                     'io/test/data/pcs.csv')
                >>> data.getPCs(file_pcs=file_pcs, nrpcs=10,
                ...                     verbose=False)
                >>> indata = input.InputData()
                >>> indata.addPCs(pcs = data.pcs,
                ...               pc_samples = data.pc_samples)
                >>> print indata.pcs.shape
                (1000, 10)
                >>> print indata.pc_samples.shape
                (1000,)

        """

        if pcs is not None:
            if pc_samples is None:
                raise MissingInput('PC sample names have to be',
                                   'specified via pcs_samples')
            if np.array(pcs).shape[0] != np.array(pc_samples).shape[0]:
                raise DataMismatch(
                    'Number of samples in pcs ({}) does',
                    'not match number of sample IDs ({}) provided'.format(
                        np.array(pcs).shape[0],
                        np.array(pc_samples).shape[0]))
            self.pcs = np.array(pcs)
            self.pc_samples = np.array(pc_samples)

        else:
            verboseprint("No principal components set", verbose=verbose)

    def subsetTraits(self, traitstring=None, traitsarray=None, verbose=True):
        r"""
        Limit analysis to specific subset of traits

	Arguments:
	    traitstring (string):
		comma-separated trait numbers (for single traits) or hyphen-
		separated trait numbers (for trait ranges) or combination of
		both for trait selection (1-based)
	    traitsarray (array-like):
		array of trait numbers for trait selection
            verbose (bool):
		 should progress messages be printed to stdout

	Returns:
            None:
                updated the following attributes of the InputData instance:

		- **self.traitsarray** (list):
		  of [`t`] trait numbers (int) to choose for analysis
            	- **self.phenotypes** (np.array):
		  reduced set of [`N` x `t`] phenotypes
            	- **self.phenotype.ID** (np.array):
		  reduced set of [`t`] phenotype IDs

	Examples:

	    .. doctest::

                >>> from limmbo.io import input
		>>> import numpy as np
		>>> from numpy.random import RandomState
		>>> random = RandomState(15)
		>>> N = 100
		>>> P = 10
		>>> pheno = random.normal(0,1, (N, P))
                >>> pheno_samples = np.array(
                ...     ['S{}'.format(x+1) for x in range(N)])
                >>> phenotype_ID = np.array(
                ...     ['ID{}'.format(x+1) for x in range(P)])
                >>> indata = input.InputData()
                >>> indata.addPhenotypes(phenotypes = pheno,
                ...                      pheno_samples = pheno_samples,
                ...                      phenotype_ID = phenotype_ID)
                >>> print indata.phenotypes.shape
                (100, 10)
                >>> print indata.phenotype_ID.shape
                (10,)
		>>> indata.subsetTraits(traitsarray=[1,2,3,6])
                >>> print indata.phenotypes.shape
                (100, 4)
                >>> print indata.phenotype_ID.shape
                (4,)
        """
        if traitstring is None and traitsarray is None:
            verboseprint('No trait subset chosen', verbose=verbose)
        else:
            if traitstring is not None and traitsarray is not None:
                verboseprint(
                    'Both traitsarray and traitstring are provided',
                    'traitstring chosen for subset selection',
                    verbose=verbose)

            elif traitstring is not None:
                verboseprint(
                    'Chose subset of {} traits'.format(traitstring),
                    verbose=verbose)
                search = re.compile(r'[^0-9,-]').search
                if bool(search(traitstring)):
                    raise FormatError(
                        'Traitstring can only contain integers',
                        '(0-9), comma (,) and hyphen (-), but {}',
                        'provided'.format(self.options.traitstring))
                traitslist = [x.split('-') for x in traitstring.split(',')]
                self.traitsarray = []
                for t in traitslist:
                    if len(t) == 1:
                        self.traitsarray.append(int(t[0]) - 1)
                    else:
                        [
                            self.traitsarray.append(x)
                            for x in range(int(t[0]) - 1, int(t[1]))
                        ]
            else:
                self.traitsarray = np.array(traitsarray)

            try:
                self.phenotypes = self.phenotypes[:, self.traitsarray]
                self.phenotype_ID = self.phenotype_ID[self.traitsarray]
            except:
                raise DataMismatch(
                    'Selected trait number {} is greater',
                    'than number of phenotypes provided {}'.format(
                        max(self.traitsarray) + 1, self.phenotypes.shape[1]))

    def commonSamples(self):
        r"""
        Get [`M]` common samples out of phenotype, relatedness and optional
        covariates with [`N`] samples (if all samples present in all datasets
        [`M`] = [`N`]) and ensure that samples are in same order.

	Arguments:
            None

	Returns:
            None:
                updated the following attributes of the InputData instance:

                - **self.phenotypes** (np.array):
                  [`M` x `P`] phenotype matrix
                - **self.pheno_samples** (np.array):
                  [`M`] sample IDs
                - **self.relatedness** (np.array):
                  [`M x M`] relatedness matrix
                - **self.relatedness_samples** (np.array):
                  [`M`] sample IDs of relatedness matrix
                - **self.covariates** (np.array):
                  [`M` x `K`] covariates matrix
                - **self.covs_samples** (np.array):
                  [`M`] sample IDs
                - **self.genotypes** (np.array):
                  [`M` x `NrSNPs`] genotypes matrix
                - **self.geno_samples** (np.array):
                  [`M`] sample IDs
                - **self.pcs** (np.array):
                  [`M` x `PCs`] principal component matrix
                - **self.pc_samples** (np.array):
                  [`M`] sample IDs

	Examples:

	    .. doctest::

                >>> from limmbo.io import input
		>>> import numpy as np
		>>> from numpy.random import RandomState
		>>> from numpy.linalg import cholesky as chol
		>>> random = RandomState(5)
                >>> P = 2
                >>> K = 4
		>>> N = 10
		>>> SNP = 1000
		>>> pheno = random.normal(0,1, (N, P))
                >>> pheno_samples = np.array(['S{}'.format(x+4)
                ...     for x in range(N)])
                >>> phenotype_ID = np.array(['ID{}'.format(x+1)
                ...     for x in range(P)])
		>>> X = (random.rand(N, SNP) < 0.3).astype(float)
                >>> relatedness = np.dot(X, X.T)/float(SNP)
                >>> relatedness_samples = np.array(['S{}'.format(x+1)
                ...     for x in range(N)])
		>>> covariates = random.normal(0,1, (N-5, K))
                >>> covs_samples = np.array(['S{}'.format(x+1)
                ...     for x in range(N-5)])
                >>> indata = input.InputData()
                >>> indata.addPhenotypes(phenotypes = pheno,
                ...                      pheno_samples = pheno_samples,
                ...                      phenotype_ID = phenotype_ID)
                >>> indata.addRelatedness(relatedness = relatedness,
                ...                  relatedness_samples = relatedness_samples)
                >>> indata.addCovariates(covariates = covariates,
                ...                      covs_samples = covs_samples)
                >>> indata.covariates.shape
                (5, 4)
                >>> indata.phenotypes.shape
                (10, 2)
                >>> indata.relatedness.shape
                (10, 10)
                >>> indata.commonSamples()
                >>> indata.covariates.shape
                (2, 4)
                >>> indata.phenotypes.shape
                (2, 2)
                >>> indata.relatedness.shape
                (2, 2)
        """
        self.samples = self.pheno_samples
        if self.relatedness is not None:
            test_pheno_relatedness = np.intersect1d(self.pheno_samples,
                                                    self.relatedness_samples)
            if len(test_pheno_relatedness) == 0:
                raise DataMismatch('No common samples between phenotypes and',
                                   'relatedness estimates')
            self.samples = test_pheno_relatedness

        if self.genotypes is not None:
            test_pheno_geno = np.intersect1d(self.pheno_samples,
                                             self.geno_samples)
            if len(test_pheno_geno) == 0:
                raise DataMismatch('No common samples between phenotypes,',
                                   'and genotypes')
            self.samples = np.intersect1d(self.samples, test_pheno_geno)

        if self.covariates is not None:
            test_pheno_covs = np.intersect1d(self.pheno_samples,
                                             self.covs_samples)
            if len(test_pheno_covs) == 0:
                raise DataMismatch('No common samples between phenotypes,',
                                   'and covariates')
            self.samples = np.intersect1d(self.samples, test_pheno_covs)

        if self.pcs is not None:
            test_pheno_pcs = np.intersect1d(self.pheno_samples,
                                            self.pcs_samples)
            if len(test_pheno_pcs) == 0:
                raise DataMismatch('No common samples between phenotypes,',
                                   'and pcs')
            self.samples = np.intersect1d(self.samples, test_pheno_pcs)

        subset_pheno_samples = np.in1d(self.pheno_samples, self.samples)
        self.pheno_samples = self.pheno_samples[subset_pheno_samples]
        self.phenotypes = self.phenotypes[subset_pheno_samples, :]
        (self.phenotypes, self.pheno_samples, samples_before,
         samples_after) = match(
             samples_ref=self.samples,
             samples_compare=self.pheno_samples,
             data_compare=self.phenotypes,
             squarematrix=False)

        if self.genotypes is not None:
            subset_geno_samples = np.in1d(self.geno_samples, self.samples)
            self.geno_samples = self.geno_samples[subset_geno_samples]
            self.genotypes = self.genotypes[subset_geno_samples, :]
            (self.genotypes, self.geno_samples, samples_before,
             samples_after) = match(
                 samples_ref=self.samples,
                 samples_compare=self.geno_samples,
                 data_compare=self.genotypes,
                 squarematrix=False)

        if self.relatedness is not None:
            subset_relatedness_samples = np.in1d(self.relatedness_samples,
                                                 self.samples)
            self.relatedness_samples = self.relatedness_samples[
                subset_relatedness_samples]
            self.relatedness = self.relatedness[subset_relatedness_samples,:]\
             [:, subset_relatedness_samples]
            (self.relatedness, self.relatedness_samples, samples_before,
             samples_after) = match(
                 samples_ref=self.samples,
                 samples_compare=self.relatedness_samples,
                 data_compare=self.relatedness,
                 squarematrix=True)

        if self.covariates is not None:
            subset_covs_samples = np.in1d(self.covs_samples, self.samples)
            self.covs_samples = self.covs_samples[subset_covs_samples]
            self.covariates = self.covariates[subset_covs_samples, :]
            self.covariates, self.covs_samples, samples_before,
            samples_after = match(
                samples_ref=self.samples,
                samples_compare=self.covs_samples,
                data_compare=self.covariates,
                squarematrix=False)

        if self.pcs is not None:
            subset_pc_samples = np.in1d(self.pc_samples, self.samples)
            self.pc_samples = self.pc_samples[subset_pc_samples]
            self.pcs = self.pcs[subset_pc_samples, :]
            self.pcs, self.pc_samples, samples_before,
            samples_after = match(
                samples_ref=self.samples,
                samples_compare=self.pc_samples,
                data_compare=self.pcs,
                squarematrix=False)

    def regress(self, regress=False, verbose=True):
        """
        Regress out covariates (optional).

        Arguments:
            regress (bool):
                if True, covariates are explanatory variables in linear model
                with phenotypes as response; residuals returned as new
                phenotype
            verbose (bool):
                should progress messages be printed to stdout

        Returns:
            None:
                updated the following attributes of the InputData instance:

                - **self.phenotypes** (np.array):
                  [`M` x `P`] phenotype matrix of residuals of linear model
                - **self.covariates**:
                  None

        Examples:

            .. doctest::

                >>> from limmbo.io import input
		>>> import numpy as np
		>>> from numpy.random import RandomState
		>>> from numpy.linalg import cholesky as chol
		>>> random = RandomState(5)
                >>> P = 5
                >>> K = 4
		>>> N = 100
		>>> pheno = random.normal(0,1, (N, P))
                >>> pheno_samples = np.array(['S{}'.format(x+1)
                ...     for x in range(N)])
                >>> phenotype_ID = np.array(['ID{}'.format(x+1)
                ...     for x in range(P)])
		>>> covariates = random.normal(0,1, (N, K))
                >>> covs_samples = np.array(['S{}'.format(x+1)
                ...     for x in range(N)])
                >>> indata = input.InputData()
                >>> indata.addPhenotypes(phenotypes = pheno,
                ...                      pheno_samples = pheno_samples,
                ...                      phenotype_ID = phenotype_ID)
                >>> indata.addCovariates(covariates = covariates,
                ...                      covs_samples = covs_samples)
                >>> indata.phenotypes[:3, :3]
                array([[ 0.44122749, -0.33087015,  2.43077119],
                       [ 1.58248112, -0.9092324 , -0.59163666],
                       [-1.19276461, -0.20487651, -0.35882895]])
                >>> indata.regress(regress=True, verbose=False)
                >>> indata.phenotypes[:3, :3]
                array([[ 0.34421705, -0.01470998,  2.25710966],
                       [ 1.69886647, -1.41756814, -0.55614649],
                       [-1.10700674, -0.66017713, -0.22201814]])
        """

        if regress:
            if np.array_equal(self.phenotypes, self.covariates):
                raise DataMismatch('Phenotype and covariate arrays are',
                                   'identical')
            verboseprint("Regress out %s" % type, verbose=verbose)
            self.phenotypes = regressOut(self.phenotypes, self.covariates)
            self.covariates = None

    def transform(self, type=None, verbose=True):
        """
        Transform phenotypes

        Arguments:
            type (string):
                transformation method for phenotype data:

                    - scale:
                      mean center, divide by sd
                    - gaussian:
                      inverse normalisation
                    - None:
                      no transformation
            verbose (bool):
                should progress messages be printed to stdout

	Returns:
            None:
                updated the following attributes of the InputData instance:

                - **self.phenotypes** (np.array):
                  [`N` x `P`] (transformed) phenotype matrix

        Examples:

            .. doctest::

                >>> from limmbo.io import input
		>>> import numpy as np
		>>> from numpy.random import RandomState
		>>> from numpy.linalg import cholesky as chol
		>>> random = RandomState(5)
                >>> P = 5
                >>> K = 4
		>>> N = 100
		>>> pheno = random.normal(0,1, (N, P))
                >>> pheno_samples = np.array(['S{}'.format(x+1)
                ...     for x in range(N)])
                >>> phenotype_ID = np.array(['ID{}'.format(x+1)
                ...     for x in range(P)])
		>>> SNP = 1000
		>>> X = (random.rand(N, SNP) < 0.3).astype(float)
                >>> relatedness = np.dot(X, X.T)/float(SNP)
                >>> relatedness_samples = np.array(['S{}'.format(x+1)
                ...     for x in range(N)])
                >>> indata = input.InputData()
                >>> indata.addPhenotypes(phenotypes = pheno,
                ...                      pheno_samples = pheno_samples,
                ...                      phenotype_ID = phenotype_ID)
                >>> indata.addRelatedness(relatedness = relatedness,
                ...                  relatedness_samples = relatedness_samples)
                >>> indata.phenotypes[:3, :3]
                array([[ 0.44122749, -0.33087015,  2.43077119],
       		       [ 1.58248112, -0.9092324 , -0.59163666],
       		       [-1.19276461, -0.20487651, -0.35882895]])
		>>> indata.transform(type='gaussian', verbose=False)
                >>> indata.phenotypes[:3, :3]
		array([[ 0.4307273 , -0.96742157,  0.96742157],
		       [ 0.96742157, -0.96742157, -0.4307273 ],
		       [-0.4307273 ,  0.4307273 ,  0.        ]])
        """

        if type == "scale":
            verboseprint("Use %s as transformation" % type, verbose=verbose)
            self.phenotypes = scale(self.phenotypes)
        elif type == "gaussian":
            verboseprint("Use %s as transformation" % type, verbose=verbose)
            self.phenotypes = np.apply_along_axis(quantile_gaussianize, 1,
                                                  self.phenotypes)
        elif type is not None:
            raise TypeError('Possible transformation methods are: scale',
                            'gaussian or None but {} provided'.format(type))
        else:
            verboseprint("Data will not be transformed", verbose=verbose)
