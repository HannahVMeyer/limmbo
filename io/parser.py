
# coding: utf-8
######################
### import modules ###
######################

import argparse

########################
### functions: input ###
########################


class DataParse(object):
    def __init__(self):
        '''
        nothing to initialize
        '''
        self.options = None

    def getArgs(self):
        # read command line parameters
        parser = argparse.ArgumentParser(description='Python2.7 script.')

        # input data
        parser.add_argument('-kf', '--file_kinship', action="store",
                            dest="file_kinship", required=False, default=None,
                            help='Kinship file')
        parser.add_argument('-pf', '--file_pheno', action="store",
                            dest="file_pheno", required=False,
                            help='HF5 pheno file')
        parser.add_argument('-cf', '--file_cov', action="store",
                            dest="file_covariates", required=False, 
                            default=None,
                            help='covariates file')
        parser.add_argument('-sf', '--file_samplelist', action="store",
                            dest="file_samplelist", required=False, 
                            default=None,
                            help=('Path to file for sample list used for'
                            'sample filtering from phenotypes and genotypes'))

        parser.add_argument('-seed', '--seed', action="store", dest="seed",
                            default=234, required=False, 
                            help=('seed used to generate bootstrap matrix'
                            '(generated in -bm mode)'), type=int)

        parser.add_argument('-sp', '--smallp', action="store", dest="p",
                            default=5, required=False,
                            help='size of matrix used for reconstruction', 
                            type=int)
        parser.add_argument('-lp', '--largeP', action="store", dest="P",
                            default=10, required=False,
                            help='size of matrix to reconstruct', type=int)
        parser.add_argument('-r', '--runs', action="store", dest="runs",
                            default=200, required=False,
                            help='Total number of bootstrap runs', type=int)
        parser.add_argument('-cache', '--cache', action="store_true",
                            dest="cache", default=False,  required=False,
                            help='should mtSet computations be cached')
        parser.add_argument('-t', '--timing', action="store_true",
                            dest="timing", default=False,  required=False,
                            help='should mtSet computations be timed')
        parser.add_argument('-minTTC', '--minCooccurrence', action="store",
                            dest="minCooccurrence", default=3,  type=int, 
                            required=False, help=('minimum count of the'
                            'pairwise sampling of any given trait pair,'
                            'default:3'))

        # settings: data transform options
        parser.add_argument('-tr', '--transform_method', action="store",
                            dest="transform", default='scale', required=False,
                            help=('Choose type of data preprocessing: scaling'
                                  'or gaussianizing'))
        parser.add_argument('-reg', '--reg_covariates', action="store_true",
                            dest="regress", required=False,
                            help=('Should covariates be regressed out?' 
                            'Default: False'))

        # settings: subset options
        parser.add_argument('-nt', '--nrTraits', action="store", dest="nt",
                            required=False, default=None, 
                            help='How many traits in subset',
                            type=int)
        parser.add_argument('-ns', '--nrSamples', action="store", dest="ns",
                            required=False, default=None,
                            help='How many samples in subset', type=int)
        parser.add_argument('-traitset', '--traitset', action="store",
                            dest="traitstring", required=False, default=None,
                            help=('which traits to choose; default: None '
                            '(=all traits)'))

        # output settings
        parser.add_argument('-of', '--output', action="store", dest="output",
                            required=True, help='Output directory')
        parser.add_argument('-v', '--verbose', action="store_true",
                            dest="verbose", required=False, default=False,
                            help=('Should analysis step description be'
                                'printed default:False'))

        self.options = parser.parse_args()
        return self
