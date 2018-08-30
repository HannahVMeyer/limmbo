from limix.util.preprocess import regressOut

from limmbo.utils.utils import verboseprint
from limmbo.utils.utils import scale
from limmbo.utils.utils import AlleleFrequencies

import pandas as pd
import numpy as np

from scipy_sugar.stats import quantile_gaussianize
from math import sqrt


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
    Class containing all datasets relevant for variance decomposition
    (phenotypes, relatedness estimates) and pre-processing steps (check for
    common samples and sample order, covariates regression and phenotype
    transformation)

    Arguments:
        verbose (bool):
            initialise verbose: should progress messages be printed to stdout
    """

    def __init__(self, verbose=True):
        self.verbose = verbose
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
        self.genotypes = None
        self.geno_samples = None
        self.genotypes_info = None
        self.Cg = None
        self.Cn = None

    def addPhenotypes(self, phenotypes, pheno_samples=None, phenotype_ID=None):
        """
        Add phenotypes, their phenotype ID and their sample IDs to
        InputData instance

        Arguments:
            phenotypes (array-like):
                [`N x `P`] phenotype matrix of `N` individuals and `P`
                phenotypes; if pandas.DataFrame with pheno_samples as index and
                phenotypes_ID as columns, pheno_samples and phenotype_ID do not
                have to specified separately.
            pheno_samples (array-like):
                [`N`] sample ID
            phenotype_ID (array-like):
                [`P`] phenotype IDs

        Returns:
            None:
                updated the following attributes of the InputData instance:

                - **self.phenotypes** (pd.DataFrame):
                  [`N` x `P`] phenotype array
                - **self.pheno_samples** (np.array):
                  [`N`] sample IDs
                - **self.phenotype_ID** (np.array):
                  [`P`] phenotype IDs


        Examples:

            .. doctest::

                >>> from limmbo.io import input
                >>> import numpy as np
                >>> import pandas as pd
                >>> pheno = np.array(((1,2),(7,1),(3,4)))
                >>> pheno_samples = ['S1','S2', 'S3']
                >>> phenotype_ID = ['ID1','ID2']
                >>> phenotypes = pd.DataFrame(pheno, index=pheno_samples,
                ...     columns = phenotype_ID)
                >>> indata = input.InputData(verbose=False)
                >>> indata.addPhenotypes(phenotypes = phenotypes)
                >>> print indata.phenotypes.shape
                (3, 2)
                >>> print indata.pheno_samples.shape
                (3,)
                >>> print indata.phenotype_ID.shape
                (2,)
        """
        if pheno_samples is None:
            try:
                self.pheno_samples = pd.DataFrame.from_dict(
                        {'id': phenotypes.index})
            except Exception:
                raise TypeError(("pheno_samples are not provided and phenotypes"
                    " has no index to retrieve pheno_samples from."))
        else:
            self.pheno_samples = pd.DataFrame.from_dict({'id': pheno_samples})
        self.pheno_samples = self.pheno_samples.astype('str')
        if phenotype_ID is None:
            try:
                self.phenotype_ID = pd.DataFrame.from_dict(
                        {'id':phenotypes.columns})
            except Exception:
                raise TypeError(("phenotype_ID are not provided and phenotypes "
                    "has no column names to retrieve phenotype_ID from."))
        else:
            self.phenotype_ID = pd.DataFrame.from_dict({'id':phenotype_ID})
        self.phenotype_ID = self.phenotype_ID.astype('str')

        if phenotypes.shape[0] != self.pheno_samples.shape[0]:
            raise DataMismatch(('Number of samples in phenotypes ({}) does not '
                ' match number of sample IDs ({}) provided').format(
                    phenotypes.shape[0], self.pheno_samples.shape[0]))
        if phenotypes.shape[1] != self.phenotype_ID.shape[0]:
            raise DataMismatch(('Number phenotypes ({}) does not match number '
                'of phenotype IDs ({}) provided').format(phenotypes.shape[1],
                    self.phenotype_ID.shape[0]))
        if len(self.pheno_samples.id) != len(set(self.pheno_samples.id)):
            raise IOError("Duplicate sample names in phenotypes")
        if len(self.phenotype_ID.id) != len(set(self.phenotype_ID.id)):
            raise IOError("Duplicate trait names in phenotypes")

        self.phenotypes = pd.DataFrame(data=np.array(phenotypes),
                    index=self.pheno_samples.id.values,
                    columns=self.phenotype_ID.id.values)

        # check for NAs
        if self.phenotypes.isna().any(axis=1).sum() != 0:
            print("Phenotypes contain NAs, samples with NA will be removed.")
            pheno_na = self.phenotypes.isna().any(axis=1)
            self.phenotypes=self.phenotypes[~pheno_na]
            self.pheno_samples=self.pheno_samples[~pheno_na.values]

    def addCovariates(self, covariates, covs_samples=None):
        """
        Add [`N` x `K`] covariate data with [`N`] samples and [`K`] covariates
        to InputData instance.

        Arguments:
            covariates (array-like):
                [`N x `K`] covariate matrix of `N` individuals and `K`
                covariates; if pandas.DataFrame with covs_samples as index,
                covs_samples do not have to specified separately.
            covs_samples (array-like):
                [`N`] sample ID

        Returns:
            None:
                updated the following attributes of the InputData instance:

                - **self.covariates** (pd.DataFrame):
                  [`N` x `K`] covariates matrix
                - **self.covs_samples** (np.array):
                  [`N`] sample IDs

        Examples:

            .. doctest::

                >>> from limmbo.io import input
                >>> import numpy as np
                >>> import pandas as pd
                >>> covariates = [(1,2,4),(1,1,6),(0,4,8)]
                >>> covs_samples = ['S1','S2', 'S3']
                >>> covariates = pd.DataFrame(covariates, index=covs_samples)
                >>> indata = input.InputData(verbose=False)
                >>> indata.addCovariates(covariates = covariates,
                ...     covs_samples = covs_samples)
                >>> print indata.covariates.shape
                (3, 3)
                >>> print indata.covs_samples.shape
                (3,)

        """
        if covs_samples is None:
            try:
                self.covs_samples = pd.DataFrame.from_dict(
                        {'id':covariates.index})
            except Exception:
                raise TypeError(("covs_samples are not provided and covariates "
                    "has no index to retrieve covs_samples from."))
        else:
            self.covs_samples = pd.DataFrame.from_dict({'id':covs_samples})
        self.covs_samples = self.covs_samples.astype('str')
        if np.array(covariates).shape[0] != self.covs_samples.shape[0]:
            raise DataMismatch(('Number of samples in covariates ({}) does not '
                'match number of sample IDs ({}) provided').format(
                    np.array(covariates).shape[0], self.covs_samples.shape[0]))
        if len(self.covs_samples.id) != len(set(self.covs_samples.id)):
            raise IOError("Duplicate sample names in covariates")
        self.covariates = pd.DataFrame(data=np.array(covariates),
                index=self.covs_samples.id.values)

        # check for NAs
        if self.covariates.isna().any(axis=1).sum() != 0:
            print("Covariates contain NAs, samples with NA will be removed.")
            cov_na = self.covariates.isna().any(axis=1)
            self.covariates=self.covariates[~cov_na]
            self.covs_samples=self.covs_samples[~cov_na.values]

    def addRelatedness(self, relatedness=None, evec_relatedness=None,
            eval_relatedness=None, relatedness_samples=None):
        """
        Add [`N` x `N`] pairwise relatedness estimates of [`N`] samples to the
        InputData instance

        Arguments:
            relatedness (array-like):
                [`N x `N`] relatedness matrix of `N` individuals;
                if pandas.DataFrame with relatedness_samples as index,
                relatedness_samples do not have to specified separately.
            relatedness_samples (array-like):
                [`N`] sample IDs
            U_relatedness (array-like):
                eigenvectors of [`N x `N`] relatedness matrix of `N` 
                individuals;
            S_relatedness (array-like):
                eigenvalues of [`N x `N`] relatedness matrix of `N` 
                individuals;

        Returns:
            None:
                updated the following attributes of the InputData instance:

                - **self.relatedness** (pd.DataFrame):
                  [`N` x `N`] relatedness matrix
                - **self.U_relatedness** (pd.DataFrame):
                  eigenvectors of [`N x `N`] relatedness matrix of `N`
                  individuals;
                - **S_relatedness** (pd.DataFrame):
                  eigenvalues of [`N x `N`] relatedness matrix of `N`
                  individuals;
                - **self.relatedness_samples** (np.array):
                  [`N`] sample IDs

        Examples:

            .. doctest::

                >>> from limmbo.io import input
                >>> import numpy
                >>> import pandas as pd
                >>> from numpy.random import RandomState
                >>> from numpy.linalg import cholesky as chol
                >>> random = RandomState(5)
                >>> N = 100
                >>> SNP = 1000
                >>> X = (random.rand(N, SNP) < 0.3).astype(float)
                >>> relatedness = numpy.dot(X, X.T)/float(SNP)
                >>> relatedness_samples = numpy.array(
                ...     ['S{}'.format(x+1) for x in range(N)])
                >>> relatedness = pd.DataFrame(relatedness,
                ...     index=relatedness_samples)
                >>> indata = input.InputData(verbose=False)
                >>> indata.addRelatedness(relatedness = relatedness)
                >>> print indata.relatedness.shape
                (100, 100)
                >>> print indata.relatedness_samples.shape
                (100,)

        """
        if all(v is None for v in [evec_relatedness,
            eval_relatedness, relatedness]):
            raise MissingInput("Neither relatedness nor eigenvectors/values of"
                    "relatedness matrix are provided")

        if relatedness_samples is None:
            if relatedness is not None:
                try:
                    self.relatedness_samples = pd.DataFrame.from_dict(
                            {'id': relatedness.index})
                except Exception:
                    raise TypeError(("relatedness_samples are not provided and "
                        "relatedness has no index to retrieve "
                        "relatedness_samples from"))
            if evec_relatedness is not None:
                try:
                    self.relatedness_samples = pd.DataFrame.from_dict(
                            {'id': evec_relatedness.index})
                except Exception:
                    raise TypeError(("relatedness_samples are not provided and "
                        "evec_relatedness has no index to retrieve "
                        "relatedness_samples from"))
        else:
            self.relatedness_samples = pd.DataFrame.from_dict(
                            {'id': relatedness_samples})
        self.relatedness_samples = self.relatedness_samples.astype('str')

        if relatedness is not None:
            rel = np.array(relatedness)

            if rel.shape[0] != rel.shape[1]:
                raise FormatError(('Relatedness has to be a square matrix, but '
                    'number of rows {} is not equal to number of '
                    'columns {}').format(rel.shape[0], rel.shape[1]))

            if not np.all(np.array(rel) - np.array(rel).T == 0):
                raise FormatError('Relatedness matrix is not symmetric')
            if not self._is_positive_definite(rel):
                raise FormatError('Relatedness matrix is not positive-semi'
                        'definite')
            if rel.shape[0] != self.relatedness_samples.shape[0]:
                raise DataMismatch(('Number of samples in relatedness ({}) does '
                    'not match number of sample IDs ({}) provided').format(
                        rel.shape[0], self.relatedness_samples.shape[0]))
            if len(self.relatedness_samples.id) != len(set(
                self.relatedness_samples.id)):
                raise IOError("Duplicate sample names in relatedness")

            self.relatedness = pd.DataFrame(relatedness,
                    index=self.relatedness_samples.id.values,
                    columns=self.relatedness_samples.id.values)
            # check for NAs
            if self.relatedness.isna().any(axis=1).sum() != 0:
                print("Relatedness matrix contains NAs, samples with NA will be "
                    "removed.")
                relatedness_samples = \
                    relatedness_samples[~self.relatedness.isna().any(axis=1).values]
                self.relatedness=self.relatedness[~self.relatedness.isna()]
            self.eval_relatedness=None
            self.evec_relatedness=None
        else:
            if any(v is None for v in [evec_relatedness, eval_relatedness]):
                raise MissingInput("Both eigenvectors and eigenvalues of"
                        "relatedness matrix have to be provided")
            self.evec_relatedness = pd.DataFrame(evec_relatedness,
                    columns=self.relatedness_samples.id)
            self.eval_relatedness = pd.DataFrame(eval_relatedness)
            self.relatedness=None



    def addGenotypes(self, genotypes, geno_samples=None,
                     genotypes_info=None, genotypes_darray=False):
        """
        Add [`N` x `NrSNP`] genotype array of [`N`] samples and [`NrSNP`]
        genotypes, [`N`] array of sample IDs and [`NrSNP` x 2] dataframe of
        genotype description to InputData instance.

        Arguments:
            genotypes (array-like):
                [`N` x `NrSNP`] genotype array of [`N`] samples and [`NrSNP`]
                genotypes; if pandas.DataFrame with geno_samples as index,
                geno_samples do not have to specified separately.
            geno_samples (array-like):
                [`N`] vector of `N` sample IDs
            genotypes_info (dataframe):
                [`NrSNPs` x 2] dataframe with columns 'chrom' and 'pos', and
                rsIDs as index
            darray (bool):
                If genotypes are dask.array set to True, False otherwise

        Returns:
            None:
                updated the following attributes of the InputData instance:

                - **self.genotypes** (pd.DataFrame):
                  [`N` x `NrSNPs`] genotype matrix
                - **self.geno_samples** (np.array):
                  [`N`] sample IDs
                - **self.genotypes_info** (pd.DataFrame):
                  [`NrSNPs` x 2] dataframe with columns 'chrom' and 'pos', and
                  rsIDs as index

        Examples:

            .. doctest::

                >>> from pkg_resources import resource_filename
                >>> from limmbo.io import reader
                >>> from limmbo.io import input
                >>> data = reader.ReadData(verbose=False)
                >>> file_geno = resource_filename(
                ...     'limmbo', 'io/test/data/genotypes.csv')
                >>> data.getGenotypes(file_genotypes=file_geno)
                >>> indata = input.InputData(verbose=False)
                >>> indata.addGenotypes(genotypes=data.genotypes,
                ...                     genotypes_info=data.genotypes_info)
                >>> indata.geno_samples[:5]
                array(['ID_1', 'ID_2', 'ID_3', 'ID_4', 'ID_5'], dtype=object)
                >>> indata.genotypes.shape
                (1000, 20)
                >>> indata.genotypes_info[:5]
                           chrom       pos
                rs1601111      3  88905003
                rs13270638     8  20286021
                rs75132935     8  76564608
                rs72668606     8  79733124
                rs55770986     7   2087823
        """
        self.genotypes_darray = genotypes_darray
        if self.genotypes_darray and geno_samples is None:
            raise MissingInput(('For genotypes in dask.array format, '
                'genotype sample IDs have to be specified via geno_samples'))
        if geno_samples is None:
            try:
                self.geno_samples = pd.DataFrame.from_dict(
                        {'id': genotypes.index})
            except Exception:
                raise TypeError(("geno_samples are not provided and genotypes "
                    "has no index to retrieve geno_samples from"))
        else:
            try:
                self.geno_samples = pd.DataFrame(data=np.array(geno_samples),
                        columns=['id'])
            except Exception:
                raise TypeError(("geno_samples are not array-like"))
        self.geno_samples = self.geno_samples.astype('str')

        if genotypes_info is None:
            raise MissingInput(('Genotype info has to be specified via '
                                'genotypes_info'))
        self.genotypes_info = genotypes_info

        if self.genotypes_darray:
            self.genotypes = genotypes
            nsamples = genotypes.shape[1]
            nsnps = genotypes.shape[0]
        else:
            self.genotypes = pd.DataFrame(genotypes,
                    index=self.geno_samples.id.values)
            nsamples = self.genotypes.shape[0]
            nsnps = self.genotypes.shape[1]

        if nsamples != self.geno_samples.shape[0]:
            raise DataMismatch(('Number of samples in genotypes ({}) does '
                'not match number of sample IDs ({}) provided').format(nsamples,
                    self.geno_samples.shape[0]))
        if nsnps != self.genotypes_info.shape[0]:
            raise DataMismatch(('Number of genotypes in genotypes ({}) does '
                'not match number of genotypes in genotypes_info ({})').format(
                    nsnps, self.genotypes_info.shape[0]))
        if len(self.geno_samples.id) != len(set(self.geno_samples.id)):
            raise IOError("Duplicate sample names in genotypes")

    def addVarianceComponents(self, Cg, Cn,):
        """
        Add [`P` x `P`] matrices of [`P`] trait covariance estimates
        of the genetic trait variance component (Cg) and the non-genetic
        (noise) variance component (Cn) to InputData instance.

        Arguments:
            Cg (array-like):
                [`P x `P`] matrix of `P` trait covariance estimates of the
                genetic trait covaraince component
            Cn (array-like):
                [`P x `P`] matrix of `P` trait covariance estimates of the
                non-genetic (noise) trait covaraince component

        Returns:
            None:
                updated the following attributes of the InputData instance:

                - **self.Cg** (np.array):
                  [`P x `P`] matrix of `P` trait covariance estimates of the
                  genetic trait covariance component
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
                >>> data = reader.ReadData(verbose=False)
                >>> file_pheno = resource_filename('limmbo',
                ...                     'io/test/data/pheno.csv')
                >>> data.getPhenotypes(file_pheno=file_pheno)
                >>> file_Cg = resource_filename('limmbo',
                ...                     'io/test/data/Cg.csv')
                >>> file_Cn = resource_filename('limmbo',
                ...                     'io/test/data/Cn.csv')
                >>> data.getVarianceComponents(file_Cg=file_Cg,
                ...                            file_Cn=file_Cn)
                >>> indata = input.InputData(verbose=False)
                >>> indata.addPhenotypes(phenotypes = data.phenotypes)
                >>> indata.addVarianceComponents(Cg = data.Cg, Cn=data.Cn)
                >>> print indata.Cg.shape
                (10, 10)
                >>> print indata.Cg.shape
                (10, 10)
        """
        if Cg is not None and Cn is not None:
            if self.phenotypes is None:
                raise FormatError(('Phenotypes have to be added before Cg/Cn '
                                   'can be added'))
            self.Cg = np.array(Cg)
            self.Cn = np.array(Cn)
            if self.Cg.shape[0] != self.Cg.shape[1]:
                raise FormatError(('Cg has to be a square matrix, but number of'
                    ' rows {} is not equal to number of columns {}').format(
                        self.Cg.shape[0], self.Cg.shape[1]))
            if not np.all(self.Cg - self.Cg.T == 0):
                raise FormatError('Cg is not symmetric')
            if not self._is_positive_definite(self.Cg):
                raise FormatError('Cg is not positive-semi definite')
            if self.Cg.shape[0] != self.phenotypes.shape[1]:
                raise DataMismatch(('Number of traits in Cg ({}) does not match'
                    ' number of traits ({}) in phenotypes').format(
                        self.Cg.shape[0], self.phenotypes.shape[1]))
            if self.Cn.shape[0] != self.Cn.shape[1]:
                raise FormatError(('Cn has to be a square matrix, but number of'
                    ' rows {} is not equal to number of columns {}').format(
                        self.Cn.shape[0], self.Cn.shape[1]))

            if not np.all(self.Cn - self.Cn.T == 0):
                raise FormatError('Cn is not symmetric')
            if not self._is_positive_definite(self.Cn):
                raise FormatError('Cn is not positive-semi definite')
            if self.Cn.shape[0] != self.phenotypes.shape[1]:
                raise DataMismatch(('Number of traits in Cn ({}) does not match'
                    ' number of traits ({}) in phenotypes').format(
                        self.Cn.shape[0], self.phenotypes.shape[1]))

    def addPCs(self, pcs, pc_samples=None):
        """
        Add [`N` x `PC`] matrix of [`PC`] principal components from the
        genotypes of [`N`] samples to InputData instance.

        Arguments:
            pcs (array-like):
                [`N x `PCs`] principal component matrix of `N` individuals and
                `PCs` principal components; if pandas.DataFrame with pc_samples
                as index, covs_samples do not have to specified separately.
            pc_samples (array-like):
                [`N`] sample IDs

        Returns:
            None:
                updated the following attributes of the InputData instance:

                - **self.pcs** (pd.DataFrame):
                  [`N` x `PCs`] principal component matrix
                - **self.pc_samples** (np.array):
                  [`N`] sample IDs

        Examples:

            .. doctest::

                >>> from pkg_resources import resource_filename
                >>> from limmbo.io import reader
                >>> from limmbo.io import input
                >>> data = reader.ReadData(verbose=False)
                >>> file_pcs = resource_filename('limmbo',
                ...                     'io/test/data/pcs.csv')
                >>> data.getPCs(file_pcs=file_pcs, nrpcs=10, delim=" ")
                >>> indata = input.InputData(verbose=False)
                >>> indata.addPCs(pcs = data.pcs)
                >>> print indata.pcs.shape
                (1000, 10)
                >>> print indata.pc_samples.shape
                (1000,)

        """
        if pc_samples is None:
            try:
                self.pc_samples = pd.DataFrame.from_dict({'id': pcs.index})
            except Exception:
                raise TypeError(("pc_samples are not provided and pcs has "
                    "no index to retrieve pc_samples from"))
        else:
            self.pc_samples = pd.DataFrame.from_dict({'id': pc_samples})
        self.pc_samples = self.pc_samples.astype('str')
        if np.array(pcs).shape[0] != np.array(self.pc_samples).shape[0]:
            raise DataMismatch(('Number of samples in pcs ({}) does not match'
                ' number of sample IDs ({}) provided').format(
                    np.array(pcs).shape[0], np.array(pc_samples).shape[0]))
        if len(self.pc_samples.id) != len(set(self.pc_samples.id)):
            raise IOError("Duplicate sample names for principle components")
        self.pcs = pd.DataFrame(pcs, index=self.pc_samples.id.values)

    def subsetTraits(self, traitlist=None):
        """
        Limit analysis to specific subset of traits

        Arguments:
            traitlist (array-like):
                array of trait numbers to select from phenotypes

        Returns:
            None:
                updated the following attributes of the InputData instance:

                - **self.traitlist** (list):
                  of [`t`] trait numbers (int) to choose for analysis
                - **self.phenotypes** (pd.DataFrame):
                  reduced set of [`N` x `t`] phenotypes
                - **self.phenotype.ID** (np.array):
                  reduced set of [`t`] phenotype IDs

        Examples:

            .. doctest::

                >>> from pkg_resources import resource_filename
                >>> from limmbo.io.reader import ReadData
                >>> from limmbo.io.input import InputData
                >>> from limmbo.io.utils import file_type
                >>> data = ReadData(verbose=False)
                >>> file_pheno = resource_filename('limmbo',
                ...                                'io/test/data/pheno.csv')
                >>> data.getPhenotypes(file_pheno=file_pheno)
                >>> traitlist = data.getTraitSubset(traitstring="1-3,5")
                >>> indata = InputData(verbose=False)
                >>> indata.addPhenotypes(phenotypes = data.phenotypes)
                >>> print indata.phenotypes.shape
                (1000, 10)
                >>> print indata.phenotype_ID.shape
                (10,)
                >>> indata.subsetTraits(traitlist=traitlist)
                >>> print indata.phenotypes.shape
                (1000, 4)
                >>> print indata.phenotype_ID.shape
                (4,)
        """
        self.traitlist = np.array(traitlist)
        if len(self.traitlist) != len(set(self.traitlist)):
            raise IOError("Duplicate trait names in traitlist")
        try:
            self.phenotypes = self.phenotypes.iloc[:, self.traitlist]
            self.phenotype_ID = self.phenotype_ID.loc[self.traitlist]
        except:
            raise DataMismatch(('Selected trait number {} is greater '
                                'than number of phenotypes provided {}'
                                ).format(max(self.traitlist) + 1,
                                         self.phenotypes.shape[1]))

    def commonSamples(self, samplelist=None):
        """
        Get [`M]` common samples out of phenotype, relatedness and optional
        covariates with [`N`] samples (if all samples present in all datasets
        [`M`] = [`N`]) and ensure that samples are in same order.

        Arguments:
            samplelist (array-like):
                array of sample IDs to select from data

        Returns:
            None:
                updated the following attributes of the InputData instance:

                - **self.phenotypes** (pd.DataFrame):
                  [`M` x `P`] phenotype matrix
                - **self.pheno_samples** (np.array):
                  [`M`] sample IDs
                - **self.relatedness** (pd.DataFrame):
                  [`M x M`] relatedness matrix
                - **self.relatedness_samples** (np.array):
                  [`M`] sample IDs of relatedness matrix
                - **self.covariates** (pd.DataFrame):
                  [`M` x `K`] covariates matrix
                - **self.covs_samples** (np.array):
                  [`M`] sample IDs
                - **self.genotypes** (pd.DataFrame):
                  [`M` x `NrSNPs`] genotypes matrix
                - **self.geno_samples** (np.array):
                  [`M`] sample IDs
                - **self.pcs** (pd.DataFrame):
                  [`M` x `PCs`] principal component matrix
                - **self.pc_samples** (np.array):
                  [`M`] sample IDs

        Examples:

            .. doctest::

                >>> from limmbo.io import input
                >>> import numpy as np
                >>> import pandas as pd
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
                >>> phenotypes = pd.DataFrame(pheno, index=pheno_samples,
                ...     columns=phenotype_ID)
                >>> X = (random.rand(N, SNP) < 0.3).astype(float)
                >>> relatedness = np.dot(X, X.T)/float(SNP)
                >>> relatedness_samples = np.array(['S{}'.format(x+1)
                ...     for x in range(N)])
                >>> covariates = random.normal(0,1, (N-2, K))
                >>> covs_samples = np.array(['S{}'.format(x+1)
                ...     for x in range(N-2)])
                >>> indata = input.InputData(verbose=False)
                >>> indata.addPhenotypes(phenotypes = pheno,
                ...                      pheno_samples = pheno_samples,
                ...                      phenotype_ID = phenotype_ID)
                >>> indata.addRelatedness(relatedness = relatedness,
                ...                  relatedness_samples = relatedness_samples)
                >>> indata.addCovariates(covariates = covariates,
                ...                      covs_samples = covs_samples)
                >>> indata.covariates.shape
                (8, 4)
                >>> indata.phenotypes.shape
                (10, 2)
                >>> indata.relatedness.shape
                (10, 10)
                >>> indata.commonSamples(samplelist=["S4", "S6", "S5"])
                >>> indata.covariates.shape
                (3, 4)
                >>> indata.phenotypes.shape
                (3, 2)
                >>> indata.relatedness.shape
                (3, 3)
        """
        self.samples = self.pheno_samples

        if self.relatedness is not None:
            pr = self.pheno_samples.id.isin(self.relatedness_samples.id)
            if len(pr.index[pr]) == 0:
                raise DataMismatch(('No common samples between phenotypes and '
                    'relatedness estimates'))
            self.samples = self.pheno_samples.loc[np.array(pd.index[pr])]

        if self.genotypes is not None:
            pg = self.pheno_samples.id.isin(self.geno_samples.id)
            pg_index = np.array(pg.index[pg])
            if len(pg_index) == 0:
                raise DataMismatch(('No common samples between phenotypes,'
                    'and genotypes'))
            spg = self.samples.id.isin(self.pheno_samples.id[pg_index])
            self.samples = self.samples.loc[np.array(spg.index[spg])]

        if self.covariates is not None:
            pc = self.pheno_samples.id.isin(self.covs_samples.id)
            pc_index = np.array(pc.index[pc])
            if len(pc_index) == 0:
                raise DataMismatch(('No common samples between phenotypes, '
                    'and covariates'))
            spc = self.samples.id.isin(self.pheno_samples.id[pc_index])
            self.samples = self.samples.loc[np.array(spc.index[spc])]

        if self.pcs is not None:
            pp = self.pheno_samples.id.isin(self.pcs_samples.id)
            pp_index = np.array(pp.index[pp])
            if len(pp_index) == 0:
                raise DataMismatch(('No common samples between phenotypes,'
                    'and pcs'))
            spp = self.samples.id.isin(self.pheno_samples.id[pp_index])
            self.samples = self.samples.loc[np.array(spp.index[spp])]

        if samplelist is not None:
            if len(samplelist) != len(set(samplelist)):
                raise IOError("Duplicate sample names in samplelist")

            ss = self.samples.id.isin(self.samplelist.id)
            ss_index = np.array(ss.index[ss])
            if len(ss_index) == 0:
                raise DataMismatch(('No common samples between samples in, '
                    'datasets and samplelist'))
            if len(ss_index) < len(samplelist):
                raise DataMismatch(('Not all Ids in samplelist are contained '
                    'in common samples from provided datasets'))
            self.samples = samplelist

        self.phenotypes = self.phenotypes.loc[self.samples.id, :]
        self.pheno_samples = self.samples

        if self.genotypes is not None:
            if not self.genotypes_darray:
                self.genotypes = self.genotypes.loc[self.samples.id, :]
            else:
                gs = self.geno_samples.id.isin(self.samples.id)
                gs_index = np.array(gs.index[gs])
                self.genotypes = self.genotypes[:, gs_index]
            self.geno_samples = self.samples
        if self.relatedness is not None:
            self.relatedness = self.relatedness.loc[self.samples.id, :]
            self.relatedness = self.relatedness[self.samples.id]
            self.relatedness_samples = self.samples
        if self.covariates is not None:
            self.covariates = self.covariates.loc[self.samples.id, :]
            self.covs_samples = self.samples
        if self.pcs is not None:
            self.pcs = self.pcs.loc[self.samples.id, :]
            self.pc_samples = self.samples

    def regress(self):
        """
        Regress out covariates (optional).

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
                >>> indata = input.InputData(verbose=False)
                >>> indata.addPhenotypes(phenotypes = pheno,
                ...                      pheno_samples = pheno_samples,
                ...                      phenotype_ID = phenotype_ID)
                >>> indata.addCovariates(covariates = covariates,
                ...                      covs_samples = covs_samples)
                >>> indata.phenotypes.values[:3, :3]
                array([[ 0.44122749, -0.33087015,  2.43077119],
                       [ 1.58248112, -0.9092324 , -0.59163666],
                       [-1.19276461, -0.20487651, -0.35882895]])
                >>> indata.regress()
                >>> indata.phenotypes.values[:3, :3]
                array([[ 0.34421705, -0.01470998,  2.25710966],
                       [ 1.69886647, -1.41756814, -0.55614649],
                       [-1.10700674, -0.66017713, -0.22201814]])
        """
        if np.array_equal(self.phenotypes, self.covariates):
            raise DataMismatch(('Phenotype and covariate arrays are '
                                'identical'))
        verboseprint('Regress covariates', verbose=self.verbose)
        phenotypes = regressOut(np.array(self.phenotypes),
                                np.array(self.covariates))
        self.phenotypes = pd.DataFrame(np.array(phenotypes),
                                       index=self.phenotypes.index,
                                       columns=self.phenotypes.columns)
        self.covariates = None

    def transform(self, transform):
        """
        Transform phenotypes

        Arguments:
            transform (string):
                transformation method for phenotype data:

                    - scale:
                      mean center, divide by sd
                    - gaussian:
                      inverse normalisation

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
                >>> indata = input.InputData(verbose=False)
                >>> indata.addPhenotypes(phenotypes = pheno,
                ...                      pheno_samples = pheno_samples,
                ...                      phenotype_ID = phenotype_ID)
                >>> indata.addRelatedness(relatedness = relatedness,
                ...                  relatedness_samples = relatedness_samples)
                >>> indata.phenotypes.values[:3, :3]
                array([[ 0.44122749, -0.33087015,  2.43077119],
                       [ 1.58248112, -0.9092324 , -0.59163666],
                       [-1.19276461, -0.20487651, -0.35882895]])
                >>> indata.transform(transform='gaussian')
                >>> indata.phenotypes.values[:3, :3]
                array([[ 0.23799988, -0.11191464,  2.05785598],
                       [ 1.41041953, -0.81365681, -0.92217818],
                       [-1.55977999,  0.01240937, -0.62091817]])
        """
        if transform == 'scale':
            verboseprint('Use %s as transformation' % transform,
                         verbose=self.verbose)
            phenotypes = scale(self.phenotypes)
            self.phenotypes = pd.DataFrame(phenotypes,
                                           index=self.phenotypes.index,
                                           columns=self.phenotypes.columns)
        elif transform == 'gaussian':
            verboseprint('Use %s as transformation' % transform,
                         verbose=self.verbose)
            phenotypes = np.apply_along_axis(quantile_gaussianize, 0,
                                             self.phenotypes)
            self.phenotypes = pd.DataFrame(phenotypes,
                                           index=self.phenotypes.index,
                                           columns=self.phenotypes.columns)
        else:
            raise TypeError(('Possible transformation methods are: scale, '
                             'and gaussian but {} provided').format(transform))

    def standardiseGenotypes(self):
        r"""
        Standardise genotypes:

        .. math::
           w_{ij} = \frac{x_{ij} -2p_i}{\sqrt{2p_i (1-p_i)}}

        where :math:`x_{ij}` is the number of copies of the reference allele
        for the :math:`i` th SNP of the :math:`j` th individual and :math:`p_i`
        is the frequency of the reference allele (as described in `(Yang et al
        2011) <http://www.pubmedcentral.nih.gov/articlerender.fcgi?artid=
        3014363&tool=pmcentrez&rendertype=abstract>`_).

        Returns:
            None:
                updated the following attributes of the InputData instance:

                - **self.genotypes_sd** (numpy array):
                  [`N` x `NrSNP`] matrix of `NrSNP` standardised genotypes for
                  `N` samples.

        Examples:

            .. doctest::

                >>> from pkg_resources import resource_filename
                >>> from limmbo.io import reader
                >>> from limmbo.io import input
                >>> from limmbo.utils.utils import makeHardCalledGenotypes
                >>> from limmbo.utils.utils import AlleleFrequencies
                >>> data = reader.ReadData(verbose=False)
                >>> file_geno = resource_filename(
                ...     'limmbo', 'io/test/data/genotypes.csv')
                >>> data.getGenotypes(file_genotypes=file_geno)
                >>> indata = input.InputData(verbose=False)
                >>> indata.addGenotypes(genotypes=data.genotypes,
                ...                     genotypes_info=data.genotypes_info)
                >>> geno_sd = indata.standardiseGenotypes()
                >>> geno_sd.iloc[:5,:3]
                             0         1       2
                ID_1 -2.201123 -2.141970 -8.9622
                ID_2 -2.201123 -2.141970 -8.9622
                ID_3 -2.201123 -2.141970 -8.9622
                ID_4  0.908627 -0.604125 -8.9622
                ID_5 -0.646248 -2.141970 -8.9622
        """
        if self.darray:
            raise TypeError("Genotype standardisation not yet implemented for"
                    "genotypes in dask.array format")
        self.genotypes_sd = np.zeros(self.genotypes.shape)
        for snp in range(self.genotypes.shape[1]):
            p, q = AlleleFrequencies(self.genotypes.iloc[:, snp])
            var_snp = sqrt(2 * p * q)
            for n in range(self.genotypes.iloc[:, snp].shape[0]):
                self.genotypes_sd[n, snp] = (np.array(self.genotypes)[n, snp] -
                                             2 * q) / var_snp

        self.genotypes_sd = pd.DataFrame(self.genotypes_sd,
                                         index=self.genotypes.index)

        return self.genotypes_sd

    def getAlleleFrequencies(self):
        """
        Compute allele frequencies of genotypes.

        Returns:
            None:
                updated the following attributes of the InputData instance:

                - **self.freqs** (pandas DataFrame):
                  [`NrSNP` x `2`] matrix of alt and ref allele frequencies;
                  index: snp IDs.

        Examples:

            .. doctest::

                >>> from pkg_resources import resource_filename
                >>> from limmbo.io import reader
                >>> from limmbo.io import input
                >>> from limmbo.utils.utils import makeHardCalledGenotypes
                >>> from limmbo.utils.utils import AlleleFrequencies
                >>> data = reader.ReadData(verbose=False)
                >>> file_geno = resource_filename(
                ...     'limmbo', 'io/test/data/genotypes.csv')
                >>> data.getGenotypes(file_genotypes=file_geno)
                >>> indata = input.InputData(verbose=False)
                >>> indata.addGenotypes(genotypes=data.genotypes,
                ...                     genotypes_info=data.genotypes_info,
                ...                     geno_samples=data.geno_samples)
                >>> freqs = indata.getAlleleFrequencies()
                >>> freqs.iloc[:5,:]
                                   p         q
                rs1601111   0.292186  0.707814
                rs13270638  0.303581  0.696419
                rs75132935  0.024295  0.975705
                rs72668606  0.119091  0.880909
                rs55770986  0.169338  0.830662
        """
        if self.darray:
            raise TypeError("Allele frequency computation not yet implemented "
                    "genotypes in dask.array format")
        verboseprint('Get allele frequencies of %s snps'.format(
            self.genotypes.shape[1]), verbose=self.verbose)
        self.freqs = np.zeros((self.genotypes.shape[1], 2))
        for snp in range(self.genotypes.shape[1]):
            self.freqs[snp, 0], self.freqs[snp, 1] = AlleleFrequencies(
                np.array(self.genotypes)[:, snp])

        self.freqs = pd.DataFrame(self.freqs, index=self.genotypes_info.index,
                                  columns=['p', 'q'])
        return self.freqs

    @staticmethod
    def _is_positive_definite(matrix):
        try:
            np.linalg.cholesky(matrix)
            return(True)
        except np.linalg.linalg.LinAlgError:
            return(False)
