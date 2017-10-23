######################
### import modules ###
######################

import argparse

########################
### functions: input ###
########################


class DataParse(object):

    def __init__(self):
        self.options = None

    def getArgs(self):
        # read command line parameters
        parser = argparse.ArgumentParser(description='Python2.7 script.')

        # input data
        parser.add_argument('-kf', '--file_kinship', action="store",
                            dest="file_relatedness", required=False, 
                            default=None, help='Path [string] to [N x (N+1)] 
                            file of kinship/relatedness matrix with [N] samples 
                            (first row: sample IDs)')
        parser.add_argument('-pf', '--file_pheno', action="store",
                            dest="file_pheno", required=False,
                            help='Path [string] to HF5 phenotype file (via 
                            limix_format_data.py) or [(N+1) x (P+1)] .csv file 
                            of [P] phenotypes with [N] samples (first column: 
                            sample IDs, first row: phenotype IDs)'))
        parser.add_argument('-cf', '--file_cov', action="store",
                            dest="file_covariates", required=False, 
                            default=None,
                            help='Path [string] to [(N+1) x C] file of 
                            covariates matrix with [N] samples and [K] 
                            covariates (first column: sample IDs, first row:
                            phenotype IDs)')
        parser.add_argument('-sf', '--file_samplelist', action="store",
                            dest="file_samplelist", required=False, 
                            default=None,
                            help='Path [string] to file with samplelist for
                            sample selection')

        parser.add_argument('-seed', '--seed', action="store", dest="seed",
                            default=234, required=False, 
                            help=('seed [int] used to generate bootstrap matrix'
                            ), type=int)

        parser.add_argument('-sp', '--smallp', action="store", dest="p",
                            default=5, required=False,
                            help='Size [int] of phenotype subsamples used for 
                            variance decomposition',  type=int)
        parser.add_argument('-lp', '--largeP', action="store", dest="P",
                            default=10, required=False,
                            help='Total number [int] of phenotypes', type=int)
        parser.add_argument('-r', '--runs', action="store", dest="runs",
                            default=None, required=False,
                            help='Total number [int] of bootstrap runs', 
                            type=int)
        parser.add_argument('-cache', '--cache', action="store_true",
                            dest="cache", default=False,  required=False,
                            help='[bool]: should mtSet computations be cached')
        parser.add_argument('-t', '--timing', action="store_true",
                            dest="timing", default=False,  required=False,
                            help='[bool]: should variance decomposition be 
                            timed')
        parser.add_argument('--minCooccurrence', action="store",
                            dest="minCooccurrence", default=3,  type=int, 
                            required=False, help=('Minimum count [int] of the'
                            'pairwise sampling of any given trait pair,'
                            'default:3'))
        parser.add_argument('-i', '--iterations', action="store",
                            dest="iterations", default=10,  required=False,
                            type=int,
                            help='Number [int] of iterations for variance 
                            decomposition attempts')
        parser.add_argument('-cpus', '--cpus', action="store",
                            dest="cpus", default=None,  required=False,
                            type=int,
                            help='Number [int] of available CPUs for
                            parallelisation of variance decomposition steps')

        # settings: data transform options
        parser.add_argument('-tr', '--transform_method', action="store",
                            dest="transform", default='scale', required=False,
                            help=('Choose type [string] of data preprocessing: 
                                scale (mean center, divide by sd) or gaussian
                                (inverse normalise)'))
        parser.add_argument('-reg', '--reg_covariates', action="store_true",
                            dest="regress", required=False,
                            help=('[bool]: should covariates be regressed out?' 
                            'Default: False'))

        # settings: subset options
        parser.add_argument('-traitset', '--traitset', action="store",
                            dest="traitstring", required=False, default=None,
                            help=('Comma- (for list of traits) or hyphen- (for
                            trait range) or comma and hyphen-separated list
                            [string] of traits (trait columns) to choose;
                            default: None (=all traits)'))

        # output settings
        parser.add_argument('-of', '--output', action="store", dest="output",
                            required=True, help='Path [string] of output 
                            directory; user needs writing permission')
        parser.add_argument('-v', '--verbose', action="store_true",
                            dest="verbose", required=False, default=False,
                            help=('[bool]: should analysis step description be'
                                'printed default: False'))

        self.options = parser.parse_args()
        return self
