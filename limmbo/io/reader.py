import scipy as sp
import pandas as pd
import numpy as np
import re

from limix.io import read_plink
from limix.io import read_gen

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
                IDs in the first column and [`P`] phenotype IDs in the first row
            delim (string): 
                delimiter of phenotype file, one of " ", ",", "\t"

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
                >>> from limmbo.io.utils import file_type
                >>> data = ReadData(verbose=False)
                >>> file_pheno = resource_filename('limmbo',
                ...                                'io/test/data/pheno.csv')
                >>> data.getPhenotypes(file_pheno=file_pheno)
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
        else:
            if file_type(file_pheno) is not "delim": 
                raise FormatError('Supplied phenotype file is neither .csv ' 
                        'nor .txt')
            verboseprint(
                "Reading phenotypes from %s" % file_pheno, verbose=self.verbose)
            try:
                self.phenotypes = pd.io.parsers.read_csv(
                    file_pheno, index_col=0, sep=delim)
            except Exception:
                raise IOError('{} could not be opened'.format(file_pheno))
            self.phenotype_ID = np.array(self.phenotypes.columns)
            self.pheno_samples = np.array(self.phenotypes.index)
            self.phenotypes = np.array(self.phenotypes)

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
                - **self.covs_samples** (np.array):
                  [`N`] sample IDs

        Examples:

            .. doctest::

                >>> from pkg_resources import resource_filename
                >>> from limmbo.io.reader import ReadData
                >>> from limmbo.io.utils import file_type
                >>> data = ReadData(verbose=False)
                >>> file_covs = resource_filename('limmbo',
                ...                               'io/test/data/covs.csv')
                >>> data.getCovariates(file_covariates=file_covs)
                >>> data.covs_samples[:3]
                array(['ID_1', 'ID_2', 'ID_3'], dtype=object)
                >>> data.covariates[:3,:3]
                array([[-1.05808516, -0.89731694,  0.18733211],
                       [ 0.28205298,  0.57994795, -0.41383724],
                       [-1.55179427, -1.70411737, -0.448364  ]])
        """

        if file_covariates is not None:
            if file_type(file_covariates) is not 'delim':
                raise FormatError('Supplied covariate file is not .csv or .txt')
            try:
                self.covariates = pd.io.parsers.read_csv(file_covariates, 
                        sep=delim)
                verboseprint("Reading covariates file", verbose=self.verbose)
            except Exception:
                raise IOError('{} could not be opened'.format(file_covariates))
            self.covs_samples = np.ravel(self.covariates.iloc[:, :1])
            self.covariates = np.array(
                self.covariates.iloc[:, 1:]).astype(float)
            # append column of 1's to adjust for mean of covariates
            self.covariates = sp.concatenate(
                [self.covariates,
                 sp.ones((self.covariates.shape[0], 1))], 1)
            self.covariates = np.array(self.covariates)
        else:
            verboseprint("No covariates set", verbose=self.verbose)
            self.covariates = None
            self.covs_samples = None

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
                - **self.relatedness_samples** (np.array):
                  [`N`] sample IDs of relatedness matrix
        Examples:

            .. doctest::

                >>> from pkg_resources import resource_filename
                >>> from limmbo.io.reader import ReadData
                >>> from limmbo.io.utils import file_type
                >>> data = ReadData(verbose=False)
                >>> file_relatedness = resource_filename('limmbo',
                ...                     'io/test/data/relatedness.csv')
                >>> data.getRelatedness(file_relatedness=file_relatedness)
                >>> data.relatedness_samples[:3]
                array(['ID_1', 'ID_2', 'ID_3'], dtype=object)
                >>> data.relatedness[:3,:3]
		array([[1.00892922e+00, 2.00758504e-04, 4.30499103e-03],
		       [2.00758504e-04, 9.98944885e-01, 4.86487318e-03],
		       [4.30499103e-03, 4.86487318e-03, 9.85787665e-01]])
        """

        if file_relatedness is None:
            raise MissingInput('No relatedness data specified')
        if file_type(file_relatedness) is not 'delim':
            raise FormatError('Supplied relatedness file is not .csv or .txt')
        try:
            self.relatedness = pd.io.parsers.read_csv(file_relatedness, 
                    sep=delim)
        except Exception:
            raise IOError('{} could not be opened'.format(file_relatedness))
        verboseprint("Reading relationship matrix", verbose=self.verbose)
        self.relatedness_samples = np.array(self.relatedness.columns)
        self.relatedness = np.array(self.relatedness).astype(float)

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
                - **self.pc_samples** (np.array):
                  [`N`] sample IDs

        Examples:

            .. doctest::

                >>> from pkg_resources import resource_filename
                >>> from limmbo.io.reader import ReadData
                >>> from limmbo.io.utils import file_type
                >>> data = ReadData(verbose=False)
                >>> file_pcs = resource_filename('limmbo',
                ...                     'io/test/data/pcs.csv')
                >>> data.getPCs(file_pcs=file_pcs, nrpcs=10, delim=" ")
                >>> data.pc_samples[:3]
                array(['ID_1', 'ID_2', 'ID_3'], dtype=object)
                >>> data.pcs[:3,:3]
                array([[-0.02632738,  0.00761785,  0.01069307],
                       [-0.05791269,  0.00538956, -0.0205066 ],
                       [-0.03619099,  0.00624196,  0.02299996]])
        """

        if file_type(file_pcs) is not 'delim':
            raise FormatError('Supplied PCA file is not .csv or .txt')

        if file_pcs is not None:
            verboseprint("Reading PCs", verbose=self.verbose)
            self.pcs = pd.io.parsers.read_csv(file_pcs, header=0, sep=delim)
            self.pc_samples = np.array(self.pcs.columns)
            self.pcs = np.array(self.pcs).astype(float)
            if nrpcs is not None:
                verboseprint(
                    "Extracting first %s pcs" % nrpcs, verbose=self.verbose)
                self.pcs = self.pcs[:, :nrpcs]
        else:
            verboseprint("No pcs set", verbose=self.verbose)
            self.pcs = None
            self.pc_samples = None

    def getGenotypes(self, file_genotypes=None, delim = ','):
        r"""
        Reads genotype file in the following formats: plink (.bed, .bim, .fam),  
        gen (.gen, .sample) or comma-separated values (.csv) file

        Arguments:
            file_geno (string):
                path to phenotype file in .plink, .gen or .csv format
                - **plink format**:

                - **gen format**:

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
                - **self.geno_samples** (np.array):
                  [`N`] sample IDs
                - **self.genotypes_info** (pd.dataframe):
                  [`NrSNPs` x 2] dataframe with columns 'chrom' and 'pos', and
                  rsIDs as index

        Examples:

            .. doctest::

                >>> from pkg_resources import resource_filename
                >>> from limmbo.io.reader import ReadData
                >>> from limmbo.io.utils import file_type
                >>> data = ReadData(verbose=False)
                >>> file_geno = resource_filename('limmbo',
                ...                                'io/test/data/genotypes.csv')
                >>> data.getGenotypes(file_genotypes=file_geno)
                >>> data.geno_samples[:5]
                array(['ID_1', 'ID_2', 'ID_3', 'ID_4', 'ID_5'], dtype=object)
                >>> data.genotypes.shape
                (1000, 20)
                >>> data.genotypes[:5,:5]
                array([[0., 1., 1., 0., 1.],
                       [0., 1., 0., 1., 1.],
                       [0., 0., 0., 0., 0.],
                       [0., 1., 0., 0., 1.],
                       [0., 0., 1., 0., 0.]])
                >>> data.genotypes_info[:5]
                            chrom       pos
                rs111647458    15  49385160
                rs67918533     15  92151569
                rs12903533     15  98887790
                rs34937778     19  18495997
                rs150270350    19  47869060

        """

        if file_genotypes is None:
            raise MissingInput('No genotypes data specified')
        if file_type(file_genotypes) not in ['delim', 'bed']:
            raise FormatError(('Supplied genotype file is neither in plink nor '
                    '.csv/.txt format'))
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

            self.geno_samples = np.array(genotypes.columns)
            self.genotypes = np.array(genotypes).astype(float).T
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
            
            self.geno_samples = np.array(fam.iid)
            self.genotypes = np.array(bed.compute()).T
            self.genotypes_info = pd.DataFrame(
                np.array([bim.chrom, bim.pos]).T,
                columns=['chrom', 'pos'],
                index=bim.snp)
                
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
                >>> file_Cg = resource_filename('limmbo',
                ...                     'io/test/data/Cg.csv')
                >>> file_Cn = resource_filename('limmbo',
                ...                     'io/test/data/Cn.csv')
                >>> data.getVarianceComponents(file_Cg=file_Cg, file_Cn=file_Cn)
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
            verboseprint(
                ("No variance components supplied, estimate variance components "
                 "before lmm test"),
                verbose=self.verbose)
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
    
    def getTraitSubset(self, traitstring = None):
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
            traitslist = [x.split('-') for x in traitstring.split(',') ]
            traitsarray = []
            for t in traitslist:
                if len(t) == 1:
                    traitsarray.append(int(t[0]) - 1)
                else:
                    [traitsarray.append(x) for x in range(int(t[0]) - 1, int(t[1])) ]
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
                try:
		    samplelist = np.array(pd.io.parsers.read_csv(
		    file_samplelist, header=None))
                    verboseprint("Read sample list from file", 
                        verbose=self.verbose)
                except Exception:
                    raise IOError('{} could not be opened'.format(
                        file_samplelist))
	    else:
                verboseprint("Read sample list", verbose=self.verbose)
		samplelist = np.array(samplelist.split(","))
	    
            verboseprint("Number of samples in sample list: %s" %
		len(samplelist), verbose=self.verbose)
            return samplelist
