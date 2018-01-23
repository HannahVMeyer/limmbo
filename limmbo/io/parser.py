import argparse

def getGWASargs():
    parser = argparse.ArgumentParser(prog="runGWAS", 
            description=('Models the association between phenotypes and '
                'genotypes, accepting additional covariates and parameters '
                'to account for population structure and relatedness between '
                'samples. Users can choose between single-trait and multi-'
                'trait models, simple linear or linear mixed model set-ups.'))
    required = parser.add_argument_group('Basic required arguments')
    required.add_argument(
        '-p',
        '--file_pheno',
        action="store",
        dest="file_pheno",
        required=False,
        help=('Path [string] to [(N+1) x (P+1)] .csv file of [P] ' 
            'phenotypes with [N] samples (first column: sample IDs, first '
            'row: phenotype IDs). Default: %(default)s'))
    required.add_argument(
        '-g', 
        '--file_geno',
        action="store",
        dest="file_geno", 
        required=False, 
        default=None,
        help=('Genotype file: either [S x N].csv file (first column: SNP id, '
             'first row: sample IDs) or plink formated genotypes '
             '(.bim/.fam/.bed). Default: %(default)s'))
    required.add_argument(
        '-o',
        '--outdir',
        action="store",
        dest="outdir",
        required=True,
        help=('Path [string] of output directory; user needs writing '
            'permission. Default: %(default)s'))
    mode = required.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        '-st',
        '--singletrait',
        action="store_true",
        dest="singletrait",
        default=False,
        help=('Set flag to conduct a single-trait association analyses'
             'Default: %(default)s'))
    
    mode.add_argument(
        '-mt',
        '--multitrait',
        action="store_true",
        dest="multitrait",
        default=False,
        help=('Set flag to conduct a multi-trait association analyses'
             'Default: %(default)s'))

    setup = required.add_mutually_exclusive_group(required=True)
    setup.add_argument(
        '-lm',
        '--lm',
        action="store_true",
        dest="lm",
        default=False,
        help=('Set flag to use a simple linear model for the association'
              'analysis'))
    setup.add_argument(
        '-lmm',
        '--lmm',
        action="store_true",
        dest="lmm",
        default=False,
        help=('Set flag to use a linear mixed model for the association'
              'analysis'))
    
    output = parser.add_argument_group('Output arguments')
    output.add_argument(
        '-n', 
        '--name',
        action="store",
        dest="name",
        help=('Name (used for output file naming). Default: '
             '%(default)s'))

    output.add_argument(
        '-v',
        '--verbose',
        action="store_true",
        dest="verbose",
        required=False,
        default=False,
        help=('[bool]: should analysis progress be displayed. '
            'Default: %(default)s'))
    
    optionalfiles = parser.add_argument_group('Optional files')
    optionalfiles.add_argument(
        '-k',
        '--file_kinship',
        action="store",
        dest="file_relatedness",
        required=False,
        default=None,
        help=('Path [string] to [N x (N+1)] file of kinship/relatedness ' 
            'matrix with [N] samples (first row: sample IDs). Required when '
            '--lmm/-lm. Default: '
            '%(default)s'))
    optionalfiles.add_argument(
        '-cgf', 
        '--file_cg', 
        action="store",
        dest="file_Cg",
        required=False,
        default=None,
        help=('Required for large phenotype sizeswhen --lmm/-lm; computed via '
             'runLiMMBo; specifies file name for genetic trait covariance '
             ' matrix (rows: traits, columns: traits). Default: %(default)s'))
    optionalfiles.add_argument(
        '-cnf', 
        '--file_cn',
        action="store",
        dest="file_Cn",
        required=False,
        default=None,
        help=('Required for large phenotype sizeswhen --lmm/-lm; computed via '
             'runLiMMBo; specifies file name for non-genetic trait covariance '
             ' matrix (rows: traits, columns: traits). Default: %(default)s'))
    optionalfiles.add_argument(
        '-pcsf', 
        '--file_pcs', 
        action="store",
        dest="file_pcs",
        required=False,
        default=None,
        help=('Path to [N x PCs] file of principal components from '
             'genotypes to be included as covariates (first column: '
             'sample IDs, first row: PC IDs); Default: %(default)s'))
    optionalfiles.add_argument(
        '-cf',
        '--file_cov',
        action="store",
        dest="file_covariates",
        required=False,
        default=None,
        help=('Path [string] to [(N+1) x C] file of covariates matrix with '
             '[N] samples and [K] covariates (first column: sample IDs, '
             'first row: phenotype IDs). Default: %(default)s'))


    optional = parser.add_argument_group('Optional association parameters')
    optional.add_argument(
        '-adjustP',
        '--adjustP',
        action="store",
        dest="adjustP",
        choices=['bonferroni', 'effective', 'None'],
        default=None,
        required=False,
        type=str,
        help=('Method to adjust single-trait p-values for'
              'multiple hypothesis testing when running'
              'multiple single-trait GWAS: bonferroni/effective number of '
              'tests `(Galwey,2009) <http://onlinelibrary.wiley.com/doi/10.1002/gepi.20408/abstract>`_'
              'Default: %(default)s'))

    optional.add_argument(
        '-nrpermutations',
        '--nrpermutations',
        action="store",
        dest="nrpermutations",
        default=None,
        required=False,
        type=int,
        help=('Number of permutations for computing empirical p-values; '
            '1/nrpermutations is maximum level of testing for '
            'significance. Default: %(default)s')
        )
    optional.add_argument(
        '-fdr',
        '--fdr',
        action="store",
        dest="fdr",
        required=False,
        default=None,
        type=float,
        help=('FDR threshold for computing empirical FDR. Default: '
            '%(default)s'))

    data = parser.add_argument_group('Optional data processing parameters')
    data.add_argument(
        '-tr',
        '--transform_method',
        action="store",
        dest="transform",
        default='scale',
        choices=['scale', 'gaussian'],
        required=False,
        help=('Choose type [string] of data preprocessing: scale (mean '
            'center, divide by sd) or gaussian (inverse normalise). ' 
            'Default: %(default)s'))
    data.add_argument(
        '-reg',
        '--reg_covariates',
        action="store_true",
        dest="regress",
        required=False,
        help=('[bool]: should covariates be regressed out? Default: '
            '%(default)s'))

    subset = parser.add_argument_group('Optional subsetting options')
    subset.add_argument(
        '-traitset',
        '--traitset',
        action="store",
        dest="traitstring",
        required=False,
        default=None,
        help=('Comma- (for list of traits) or hyphen- (for trait range) or '
        'comma and hyphen-separated list [string] of traits (trait '
        'columns) to choose; default: None (=all traits). Default: '
        '%(default)s'))
    
    subset.add_argument(
        '-nrpcs',
        '--nrpcs',
        action="store",
        dest="nrpcs",
        required=False,
        default=10,
        help='First PCs to chose. Default: %(default)s',
        type= int)

    subset.add_argument(
        '-sf',
        '--file_samplelist',
        action="store",
        dest="file_samplelist",
        required=False,
        default=None,
        help=('Path [string] to file with samplelist for sample '
            'selection. Default: %(default)s'))

    plotting = parser.add_argument_group('Plot arguments', 
            'Arguments for depicting GWAS results as manhattan plot')
    plotting.add_argument(
        '-colourS',
        '--colourS',
        action="store",
        dest="colourS",
        required=False,
        default='DarkBlue',
        help=('Colour of significant points in manhattan plot'))
    plotting.add_argument(
        '-colourNS',
        '--colourNS',
        action="store",
        dest="colourNS",
        required=False,
        default='Orange',
        help=('Colour of non-significant points in manhattan plot'))
    plotting.add_argument(
        '-alphaNS',
        '--alphaNS',
        action="store",
        dest="alphaNS",
        required=False,
        default=0.05,
        help=('Transparency of non-significant points in manhattan plot'))

    version = parser.add_argument_group('Version')
    version.add_argument('--version', action='version', version='%(prog)s 0.1.0')

    return parser

def getVarianceEstimationArgs():
    parser = argparse.ArgumentParser(prog="runVarianceEstimation",
            description=('Estimates the genetic and non-genetic trait' 
                'covariance matrix parameters of a linear mixed model '
                'with random genetic and non-genetic effect via a '
                'bootstrapping-based approach.'))
    required = parser.add_argument_group('Basic required arguments')
    required.add_argument(
        '-p',
        '--file_pheno',
        action="store",
        dest="file_pheno",
        required=False,
        help=('Path [string] to [(N+1) x (P+1)] .csv file of [P] '
            'phenotypes with [N] samples (first column: sample IDs, first '
            'row: phenotype IDs). Default: %(default)s'))
    required.add_argument(
        '-k',
        '--file_kinship',
        action="store",
        dest="file_relatedness",
        required=False,
        default=None,
        help=('Path [string] to [N x (N+1)] file of kinship/relatedness '
            'matrix with [N] samples (first row: sample IDs). Required when '
            '--lmm/-lm. Default: '
            '%(default)s'))
    required.add_argument(
        '-o',
        '--outdir',
        action="store",
        dest="outdir",
        required=True,
        help=('Path [string] of output directory; user needs writing '
            'permission. Default: %(default)s'))

    optionalfiles = parser.add_argument_group('Optional files')
    optionalfiles.add_argument(
        '-c',
        '--file_cov',
        action="store",
        dest="file_covariates",
        required=False,
        default=None,
        help=('Path [string] to [(N+1) x C] file of covariates matrix with '
             '[N] samples and [K] covariates (first column: sample IDs, '
             'first row: phenotype IDs). Default: %(default)s'))

    limmbo = parser.add_argument_group('Bootstrapping parameters')
    limmbo.add_argument(
        '-seed',
        '--seed',
        action="store",
        dest="seed",
        default=234,
        required=False,
        help=('seed [int] used to generate bootstrap matrix. Default: '
            '%(default)s'),
        type=int)

    limmbo.add_argument(
        '-sp',
        '--smallp',
        action="store",
        dest="S",
        default=None,
        required=True,
        help=('Size [int] of phenotype subsamples used for variance '
            'decomposition. Default: %(default)s'),
        type=int)
    limmbo.add_argument(
        '-r',
        '--runs',
        action="store",
        dest="runs",
        default=None,
        required=False,
        help='Total number [int] of bootstrap runs. Default: %(default)s',
        type=int)
    limmbo.add_argument(
        '-t',
        '--timing',
        action="store_true",
        dest="timing",
        default=False,
        required=False,
        help=('[bool]: should variance decomposition be timed. Default: '
            '%(default)s'))
    limmbo.add_argument(
        '--minCooccurrence',
        action="store",
        dest="minCooccurrence",
        default=3,
        type=int,
        required=False,
        help=('Minimum count [int] of the pairwise sampling of any given '
            'trait pair. Default: %(default)s'))
    limmbo.add_argument(
        '-i',
        '--iterations',
        action="store",
        dest="iterations",
        default=10,
        required=False,
        type=int,
        help=('Number [int] of iterations for variance decomposition '
            'attempts. Default: %(default)s'))
    limmbo.add_argument(
        '-cpus',
        '--cpus',
        action="store",
        dest="cpus",
        default=None,
        required=False,
        type=int,
        help=('Number [int] of available CPUs for parallelisation of '
            'variance decomposition steps. Default: %(default)s'))
    
    data = parser.add_argument_group('Optional data processing parameters')
    data.add_argument(
        '-tr',
        '--transform_method',
        action="store",
        dest="transform",
        default='scale',
        choices=['scale', 'gaussian'],
        required=False,
        help=('Choose type [string] of data preprocessing: scale (mean '
            'center, divide by sd) or gaussian (inverse normalise). '
            'Default: %(default)s'))
    data.add_argument(
        '-reg',
        '--reg_covariates',
        action="store_true",
        dest="regress",
        required=False,
        help=('[bool]: should covariates be regressed out? Default: '
            '%(default)s'))


    subset = parser.add_argument_group('Optional subsetting options')
    subset.add_argument(
        '-traitset',
        '--traitset',
        action="store",
        dest="traitstring",
        required=False,
        default=None,
        help=('Comma- (for list of traits) or hyphen- (for trait range) or '
        'comma and hyphen-separated list [string] of traits (trait '
        'columns) to choose; default: None (=all traits). Default: '
        '%(default)s'))

    subset.add_argument(
        '-sf',
        '--file_samplelist',
        action="store",
        dest="file_samplelist",
        required=False,
        default=None,
        help=('Path [string] to file with samplelist for sample '
            'selection. Default: %(default)s'))

    output = parser.add_argument_group('Output arguments')
    output.add_argument(
        '-dontSaveIntermediate',
        '--dontSaveIntermediate',
        action="store_false",
        dest="intermediate",
        default=True,
        required=False,
        help=('Set to suppress saving intermediate variance components. '
            'Default: %(default)s'))
    
    output.add_argument(
        '-v',
        '--verbose',
        action="store_true",
        dest="verbose",
        required=False,
        default=False,
        help=('[bool]: should analysis step description be printed. '
            'Default: %(default)s'))
    
    version = parser.add_argument_group('Version')
    version.add_argument('--version', action='version', version='%(prog)s 0.1.0')

    return parser
