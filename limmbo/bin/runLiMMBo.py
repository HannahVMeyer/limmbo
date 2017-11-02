######################
### import modules ###
######################

import h5py

import sys
import os

from limmbo.io.parser import DataParse
from limmbo.io.input import DataInput
from limmbo.core.vdbootstrap import DataLimmbo

#################
### functions ###
#################


def entry_point():

    # initiate DataParse object
    dataparse = DataParse()
    dataparse.getArgs()

    # initiate Data objects
    datainput = DataInput(dataparse.options)

    ### running analyses ###
    datainput.getPhenotypes()
    datainput.getRelatedness()
    #datainput.commonSamples()
    datainput.subsetTraits()
    datainput.getCovariates()
    datainput.regress_and_transform()
    datalimmbo = DataLimmbo(datainput=datainput, options=dataparse.options)
    resultsQ = datalimmbo.sampleCovarianceMatricesPP()
    datalimmbo.combineBootstrap(resultsQ)


#############
### main  ###
#############

if __name__ == "__main__":
    main()
