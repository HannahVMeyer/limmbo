######################
### import modules ###
######################

import h5py
import pdb
import sys
import os
sys.path = ['/homes/hannah/LiMMBo/limmbo'] + \
['/homes/hannah/bin/python_modules/mtSet'] + sys.path

import unittest as unittest
import numpy as np

from limmbo.io.parser import DataParse
from limmbo.io.input import DataInput
from limmbo.io.input import MissingInput
from limmbo.io.input import DataMismatch
from limmbo.io.input import FormatError
from limmbo.core.vdbootstrap import DataLimmbo

#################
### functions ###
#################

class Options(object):
    def __init__(self, file_pheno = "test"):
        self.file_pheno = file_pheno
        self.verbose = False

class Input(unittest.TestCase):

    def setUp(self):
        self.datainput = DataInput()
        self.phenotypes = np.array(((1,2),(1,3)))
        self.pheno_samples = np.array(('S1','S2'))
        self.phenotype_ID = np.array(('ID2','ID2'))
    
    def test_getPhenotypes_missing_phenotypes(self):
        with self.assertRaises(MissingInput):
	    self.datainput.getPhenotypes()
    
    def test_getPhenotypes_missing_phenotype_ID(self):
        with self.assertRaises(MissingInput):
	    self.datainput.getPhenotypes(phenotypes=self.phenotypes,
                                         pheno_samples=self.pheno_samples)
    
    def test_getPhenotypes_missing_pheno_samples(self):
        with self.assertRaises(MissingInput):
	    self.datainput.getPhenotypes(phenotypes=self.phenotypes,
                                         phenotype_ID=self.phenotype_ID)
    
    def test_getPhenotypes_mismatch_phenotype_ID_phenotype_dimension(self):
        pheno_ID=np.array(('ID1', 'ID2', 'ID3'))
        with self.assertRaises(DataMismatch):
	    self.datainput.getPhenotypes(phenotypes=self.phenotypes,
                                         phenotype_ID=pheno_ID,
                                         pheno_samples=self.pheno_samples)
    
    def test_getPhenotypes_mismatch_pheno_samples_phenotype_dimension(self):
        pheno_samples=np.array(('S1', 'S2', 'S3'))
        with self.assertRaises(DataMismatch):
	    self.datainput.getPhenotypes(phenotypes=self.phenotypes,
                                         phenotype_ID=self.phenotype_ID,
                                         pheno_samples=pheno_samples)
    
    def test_getPhenotypes_fromOptions_wrongFileFormat(self):
        self.options = Options()
        self.datainputFromOptions = DataInput(self.options)
        with self.assertRaises(FormatError):
	    self.datainputFromOptions.getPhenotypes()
    
    def test_getPhenotypes_fromOptions_FileNonExisting(self):
        self.options = Options(file_pheno="test.h5")
        self.datainputFromOptions = DataInput(self.options)
        with self.assertRaises(IOError):
	    self.datainputFromOptions.getPhenotypes()


def soon():
    ### make tests for ###
    datainput.getRelatedness()
    datainput.commonSamples()
    datainput.subsetTraits()
    datainput.getCovariates()
    datainput.regress_and_transform()

    datalimmbo = DataLimmbo(datainput=datainput, options=dataparse.options)
    resultsQ = datalimmbo.sampleCovarianceMatricesPP()
    datalimmbo.combineBootstrap(resultsQ)

if __name__ == '__main__':
    unittest.main()
