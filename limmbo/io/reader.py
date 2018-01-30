import scipy as sp
import pandas as pd
import numpy as np
import re

from limix.io import read_plink

from limmbo.utils.utils import verboseprint
from limmbo.io.utils import file_type


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
        self.genotypes = None
        self.geno_samples = None
        self.genotypes_info = None
        self.pcs = None
        self.pc_samples = None
        self.snps = None
        self.geno_samples = None
        self.position = None
        self.Cg = None
        self.Cn = None

    def getPhenotypes(self, file_pheno=None, delim=","):
        r"""
        Reads [`N` x `P`] phenotype file; file ending must be either .txt or
        .csv

        Arguments:
            file_pheno (string):
                path to [(`N`+1) x (`P`+1)] phenotype file with: [`N`] sample
                IDs in the first column and [`P`] phenotype IDs in the first
                row
            delim (string):
                delimiter of phenotype file, one of " ", ",", "\t"

        Returns:
            None:
                updated the following attributes of the ReadData instance:

                - **self.phenotypes** (np.array):
                  [`N` x `P`] phenotype matrix

        Examples:

            .. doctest::

                >>> from pkg_resources import resource_filename
                >>> from limmbo.io.reader import ReadData
                >>> from limmbo.io.utils import file_type
                >>> data = ReadData(verbose=False)
                >>> file_pheno = resource_filename('limmbo',
                ...                                'io/test/data/pheno.csv')
                >>> data.getPhenotypes(file_pheno=file_pheno)
                >>> data.phenotypes.index[:3]
                Index([u'ID_1', u'ID_2', u'ID_3'], dtype='object')
                >>> data.phenotypes.columns[:3]
                Index([u'trait_1', u'trait_2', u'trait_3'], dtype='object')
        """

        if file_pheno is None:
            raise MissingInput('No phenotype file specified')
        else:
            if file_type(file_pheno) is not "delim":
                raise FormatError('Supplied phenotype file is neither .csv '
                                  'nor .txt')
            verboseprint(
                "Reading phenotypes from %s" % file_pheno,
                verbose=self.verbose)
            try:
                self.phenotypes = pd.io.parsers.read_csv(
                    file_pheno, index_col=0, sep=delim)
            except Exception:
                raise IOError('{} could not be opened'.format(file_pheno))

    def getCovariates(self, file_covariates=None, delim=','):
        r"""
        Reads [`N` x `K`] covariate matrix with [`N`] samples and [`K`]
        covariates.

        Arguments:
            file_covariates (string):
                [`N` x (`K` +1)] covariates file with [`N`] sample IDs in the
                first column
            delim (string):
                delimiter of covariates file, one of " ", ",", "\t"

        Returns:
            None:
                updated the following attributes of the ReadData instance:

                - **self.covariates** (np.array):
                  [`N` x `K`] covariates matrix

        Examples:

            .. doctest::

                >>> from pkg_resources import resource_filename
                >>> from limmbo.io.reader import ReadData
                >>> from limmbo.io.utils import file_type
                >>> data = ReadData(verbose=False)
                >>> file_covs = resource_filename('limmbo',
                ...                               'io/test/data/covs.csv')
                >>> data.getCovariates(file_covariates=file_covs)
                >>> data.covariates.index[:3]
                Index([u'ID_1', u'ID_2', u'ID_3'], dtype='object')
        """

        if file_covariates is not None:
            if file_type(file_covariates) is not 'delim':
                raise FormatError(
                    'Supplied covariate file is not .csv or .txt')
            try:
                self.covariates = pd.io.parsers.read_csv(file_covariates,
                                                         sep=delim,
                                                         index_col=0)
                verboseprint("Reading covariates file", verbose=self.verbose)
            except Exception:
                raise IOError('{} could not be opened'.format(file_covariates))
            # append column of 1's to adjust for mean of covariates
            self.covariates = pd.concat([self.covariates, pd.DataFrame(sp.ones(
                self.covariates.shape[0]), index=self.covariates.index)],
                axis=1)
        else:
            verboseprint("No covariates set", verbose=self.verbose)
            self.covariates = None

    def getRelatedness(self, file_relatedness, delim=","):
        """
        Read file of [`N` x `N`] pairwise relatedness estimates of [`N`]
        samples.

        Arguments:
            file_relatedness (string):
                [(`N` + `1`) x N] .csv file with: [`N`] sample IDs in the first
                row
            delim (string):
                delimiter of covariate file, one of " ", ",", "\t"

        Returns:
            None:
                updated the following attributes of the ReadData instance:

                - **self.relatedness** (np.array):
                  [`N` x `N`] relatedness matrix
        Examples:

            .. doctest::

                >>> from pkg_resources import resource_filename
                >>> from limmbo.io.reader import ReadData
                >>> from limmbo.io.utils import file_type
                >>> data = ReadData(verbose=False)
                >>> file_relatedness = resource_filename('limmbo',
                ...                     'io/test/data/relatedness.csv')
                >>> data.getRelatedness(file_relatedness=file_relatedness)
                >>> data.relatedness.index[:3]
                Index([u'ID_1', u'ID_2', u'ID_3'], dtype='object')
                >>> data.relatedness.columns[:3]
                Index([u'ID_1', u'ID_2', u'ID_3'], dtype='object')
        """

        if file_relatedness is None:
            raise MissingInput('No relatedness data specified')
        if file_type(file_relatedness) is not 'delim':
            raise FormatError('Supplied relatedness file is not .csv or .txt')
        try:
            self.relatedness = pd.io.parsers.read_csv(file_relatedness,
                                                      sep=delim)
            self.relatedness.index = self.relatedness.columns
        except Exception:
            raise IOError('{} could not be opened'.format(file_relatedness))
        verboseprint("Reading relationship matrix", verbose=self.verbose)

    def getPCs(self, file_pcs=None, nrpcs=None, delim=","):
        r"""
        Reads file with [`N` x `PC`] matrix of [`PC`] principal components from
        the genotypes of [`N`] samples.

        Arguments:
            file_pcs (string):
                [`N` x (`PC` +1)] PCA file with [`N`] sample IDs in the first
                column
            delim (string):
                delimiter of PCA file, one of " ", ",", "\t"
            nrpcs (integer):
                Number of PCs to use (uses first nrpcs  principal components)

        Returns:
            None:
                updated the following attributes of the ReadData instance:

                - **self.pcs** (np.array):
                  [`N` x `PC`] principal component matrix

        Examples:

            .. doctest::

                >>> from pkg_resources import resource_filename
                >>> from limmbo.io.reader import ReadData
                >>> from limmbo.io.utils import file_type
                >>> data = ReadData(verbose=False)
                >>> file_pcs = resource_filename('limmbo',
                ...                     'io/test/data/pcs.csv')
                >>> data.getPCs(file_pcs=file_pcs, nrpcs=10, delim=" ")
                >>> data.pcs.index[:3]
                Index([u'ID_1', u'ID_2', u'ID_3'], dtype='object', name=0)
                >>> data.pcs.values[:3,:3]
                array([[-0.02632738, -0.05791269, -0.03619099],
                       [ 0.00761785,  0.00538956,  0.00624196],
                       [ 0.01069307, -0.0205066 ,  0.02299996]])
        """

        if file_type(file_pcs) is not 'delim':
            raise FormatError('Supplied PCA file is not .csv or .txt')

        if file_pcs is not None:
            verboseprint("Reading PCs", verbose=self.verbose)
            self.pcs = pd.io.parsers.read_csv(file_pcs, index_col=0,
                                              sep=delim, header=None)
            if nrpcs is not None:
                verboseprint(
                    "Extracting first %s pcs" % nrpcs, verbose=self.verbose)
                self.pcs = self.pcs.iloc[:, :nrpcs]
        else:
            verboseprint("No pcs set", verbose=self.verbose)
            self.pcs = None

    def getGenotypes(self, file_genotypes=None, delim=','):
        r"""
        Reads genotype file in the following formats: plink (.bed, .bim, .fam),
        gen (.gen, .sample) or comma-separated values (.csv) file.

        Arguments:
            file_geno (string):
                path to phenotype file in .plink or .csv format
                - **plink format**:

                  as specified in the plink `user manual <https://www.cog-
                  genomics.org/plink/1.9/input>`_, binary plink format with
                  .bed, .fam and .bim file.

                - **.csv format**:

                  - [(`NrSNP` + 1) x (`N`+1)] .csv file with: [`N`] sample IDs
                    in the first row and [`NrSNP`] genotype IDs in the first
                    column
                  - sample IDs should be of type: chrom-pos-rsID for instance
                    22-50714616-rs6010226
            delim (string):
                delimiter of genotype file (when text format), one of " ", ",",
                "\t"

        Returns:
            None:
                updated the following attributes of the ReadData instance:

                - **self.genotypes** (np.array):
                  [`N` x `NrSNPs`] genotype matrix
                - **self.genotypes_info** (pd.dataframe):
                  [`NrSNPs` x 2] dataframe with columns 'chrom' and 'pos', and
                  rsIDs as index

        Examples:

            .. doctest::

                >>> from pkg_resources import resource_filename
                >>> from limmbo.io import reader
                >>> from limmbo.io.utils import file_type
                >>> data = reader.ReadData(verbose=False)
                >>> # Read genotypes in delim-format
                >>> file_geno = resource_filename('limmbo',
                ...     'io/test/data/genotypes.csv')
                >>> data.getGenotypes(file_genotypes=file_geno)
                >>> data.genotypes.index[:4]
                Index([u'ID_1', u'ID_2', u'ID_3', u'ID_4'], dtype='object')
                >>> data.genotypes.shape
                (1000, 20)
                >>> data.genotypes_info[:5]
                           chrom       pos
                rs1601111      3  88905003
                rs13270638     8  20286021
                rs75132935     8  76564608
                rs72668606     8  79733124
                rs55770986     7   2087823
                >>> ### read genotypes in plink format
                >>> file_geno = resource_filename('limmbo',
                ...     'io/test/data/genotypes')
                >>> data.getGenotypes(file_genotypes=file_geno)
                >>> data.genotypes_info[:5]
                           chrom       pos
                rs1601111      3  88905003
                rs13270638     8  20286021
                rs75132935     8  76564608
                rs72668606     8  79733124
                rs55770986     7   2087823
        """

        if file_genotypes is None:
            raise MissingInput('No genotypes data specified')
        if file_type(file_genotypes) not in ['delim', 'bed']:
            raise FormatError(('Supplied genotype file is neither in plink '
                               'nor .csv/.txt format'))
        verboseprint("Reading genotypes from %s" % file_genotypes,
                     verbose=self.verbose)

        if file_type(file_genotypes) is 'delim':
            try:
                genotypes = pd.io.parsers.read_csv(
                    file_genotypes, sep=delim, index_col=0, header=0)
            except Exception:
                raise IOError('{} could not be opened'.format(file_genotypes))

            info = np.array(genotypes.index)

            genotypes_info = []
            snp_ID = []
            for id in range(info.shape[0]):
                split = np.array(info[id].split('-'))
                snp_ID.append(split[2])
                genotypes_info.append(split[[0, 1]])

            self.genotypes = genotypes.astype(float).T
            self.genotypes_info = pd.DataFrame(
                np.array(genotypes_info),
                columns=['chrom', 'pos'],
                index=snp_ID)

        if file_type(file_genotypes) is 'bed':
            try:
                (bim, fam, bed) = read_plink(file_genotypes,
                                             verbose=self.verbose)
            except Exception:
                raise IOError('{} could not be opened'.format(file_genotypes))

            self.genotypes = pd.DataFrame(bed.compute()).astype(float).T
            self.genotypes.index = fam.iid
            self.genotypes_info = pd.DataFrame(
                np.array([bim.chrom, bim.pos]).T,
                columns=['chrom', 'pos'],
                index=bim.snp)
            self.genotypes_info.index.name = None

    def getVarianceComponents(self, file_Cg=None, file_Cn=None, delim_cg=",",
                              delim_cn=","):
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
                >>> file_Cg = resource_filename(
                ...     'limmbo', 'io/test/data/Cg.csv')
                >>> file_Cn = resource_filename(
                ...     'limmbo', 'io/test/data/Cn.csv')
                >>> data.getVarianceComponents(file_Cg=file_Cg,
                ...                            file_Cn=file_Cn)
                >>> data.Cg.shape
                (10, 10)
                >>> data.Cn.shape
                (10, 10)
                >>> data.Cg[:3,:3]
                array([[ 0.45446454, -0.21084613,  0.01440468],
                       [-0.21084613,  0.11443656,  0.01250233],
                       [ 0.01440468,  0.01250233,  0.02347906]])
                >>> data.Cn[:3,:3]
                array([[ 0.53654803, -0.14392748, -0.45483001],
                       [-0.14392748,  0.88793093,  0.30539822],
                       [-0.45483001,  0.30539822,  0.97785614]])
        """

        if file_Cg is None and file_Cn is None:
            verboseprint(
                ("No variance components supplied, estimate variance "
                 "components before lmm test"), verbose=self.verbose)
        elif file_Cg is None or file_Cn is None:
            raise IOError('Both variant components need to be supplied:',
                          'Cg is %s and Cn is %s') % (file_Cg, file_Cn)
        else:
            try:
                self.Cg = np.array(
                    pd.io.parsers.read_csv(file_Cg, header=None, sep=delim_cg))
            except Exception:
                raise IOError('{} could not be opened'.format(file_Cg))
            try:
                self.Cn = np.array(
                    pd.io.parsers.read_csv(file_Cn, header=None, sep=delim_cn))
            except Exception:
                raise IOError('{} could not be opened'.format(file_Cn))

    def getTraitSubset(self, traitstring=None):
        """
        Limit analysis to specific subset of traits

        Arguments:
            traitstring (string):
                comma-separated trait numbers (for single traits) or hyphen-
                separated trait numbers (for trait ranges) or combination of
                both for trait selection (1-based)

        Returns:
            (numpy array)
                array containing list of trait IDs

        Examples:

            .. doctest::

                >>> from limmbo.io import reader
                >>> data = reader.ReadData(verbose=False)
                >>> traitlist = data.getTraitSubset("1,3,5,7-10")
                >>> print traitlist
                [0 2 4 6 7 8 9]
        """
        if traitstring is None:
            verboseprint('No trait subset chosen', verbose=self.verbose)
        else:
            verboseprint('Chose subset of {} traits'.format(traitstring),
                         verbose=self.verbose)
            search = re.compile('[^0-9,-]').search
            if bool(search(traitstring)):
                raise FormatError(('Traitstring can only contain integers '
                                   '(0-9), comma (,) and hyphen (-), but {}'
                                   'provided').format(traitstring))
            traitslist = [x.split('-') for x in traitstring.split(',')]
            traitsarray = []
            for t in traitslist:
                if len(t) == 1:
                    traitsarray.append(int(t[0]) - 1)
                else:
                    [traitsarray.append(x)
                     for x in range(int(t[0]) - 1, int(t[1]))]
            return np.array(traitsarray)

    def getSampleSubset(self, file_samplelist=None, samplelist=None):
        r"""
        Read file or string with subset of sample IDs to analyse.

        Arguments:
            file_samplelist (string):
                "path/to/file_samplelist": file contains subset sample IDs with
                one ID per line, no header.
            samplestring (string):
                comma-separated list of sample IDs e.g. "ID1,ID2,ID5,ID10".

        Returns:
            (numpy array)
                array containing list of sample IDs
        """
        if file_samplelist is not None and samplelist is not None:
            raise IOError("Only one of file_samplelist or samplelist can "
                          "be specified")
        if file_samplelist is not None or samplelist is not None:
            if file_samplelist is not None:
                verboseprint("Read sample list from file",
                             verbose=self.verbose)
                try:
                    samplelist = pd.io.parsers.read_csv(
                        file_samplelist, sep=" ", header=None, index_col=0)
                except Exception:
                    raise IOError('{} could not be opened'.format(
                        file_samplelist))
                samplelist = samplelist.index
            else:
                verboseprint("Split sample string", verbose=self.verbose)
                samplelist = samplelist.split(",")

            verboseprint("Number of samples in sample list: %s" %
                         len(samplelist), verbose=self.verbose)
            return samplelist
