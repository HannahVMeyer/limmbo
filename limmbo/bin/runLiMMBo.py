
# coding: utf-8
######################
### import modules ###
######################

import h5py

import sys
import os
path_abs = os.path.dirname(os.path.abspath(sys.argv[0]))
path_limmbo = os.path.join(path_abs,'../..')
sys.path.append(path_limmbo)
#sys.path.append('/homes/hannah/bin/python_modules')
#sys.path.append('/homes/hannah/LiMMBo')
#sys.path.append(
 #   '/homes/hannah/software/python2.7.8/lib/python2.7/site-packages')

from limmbo.io.parser import DataParse
from limmbo.io.input import DataInput
from limmbo.core.vdbootstrap import DataLimmbo


import pdb


def main():

    # initiate DataParse object
    dataparse = DataParse()
    dataparse.getArgs()

    # initiate Data objects
    datainput = DataInput(dataparse.options)

    ### running analyses ###
    datainput.getPhenotypes()
    datainput.getKinship()
    datainput.commonSamples()
    datainput.subsetTraits()
    datainput.getCovariates()
    datainput.regress_and_transform()
    datalimmbo = DataLimmbo(datainput=datainput, options=dataparse.options)
    resultsQ = datalimmbo.sampleCovarianceMatrices()
    datalimmbo.combineBootstrap(resultsQ)


if __name__ == "__main__":
    main()
