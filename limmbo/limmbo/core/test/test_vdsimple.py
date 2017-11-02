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
import pandas as pd

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
    def __init__(self, verbose=False, output="test"):
        self.verbose = verbose
        self.output = output

class Input(unittest.TestCase):

    def setUp(self):
        self.datainput = DataInput()
        self.phenotypes = np.array(((1,2),(1,3)))
        self.pheno_samples = np.array(('S1','S2'))
        self.phenotype_ID = np.array(('ID2','ID2'))
    
        self.covariates = np.array((1,2))
        self.covs_samples = np.array(('S1','S2'))
        
        self.relatedness = np.array(((1,2),(2,1)))
        self.relatedness_samples = np.array(('S1','S2'))


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
        pheno_ID=np.array(('ID1', 'ID2', 'ID3', 'ID4'))
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

    def test_getPhenotypes_phenotypes_type(self):
        phenotypes=pd.DataFrame([(1,2),(2,3)])
        with self.assertRaises(TypeError):
	    self.datainput.getPhenotypes(phenotypes=phenotypes,
                                         pheno_samples=self.pheno_samples,
                                         phenotype_ID=self.phenotype_ID)

    def test_getPhenotypes_phenotype_ID_type(self):
        phenotype_ID=('ID1','ID2')
        with self.assertRaises(TypeError):
	    self.datainput.getPhenotypes(phenotypes=self.phenotypes,
                                         pheno_samples=self.pheno_samples,
                                         phenotype_ID=phenotype_ID)
    def test_getPhenotypes_pheno_samples_type(self):
        pheno_samples=['S1','S2']
        with self.assertRaises(TypeError):
	    self.datainput.getPhenotypes(phenotypes=self.phenotypes,
                                         pheno_samples=pheno_samples,
                                         phenotype_ID=self.phenotype_ID)

    def test_getCovariates_missing_covs_samples(self):
        with self.assertRaises(MissingInput):
	    self.datainput.getCovariates(covariates=self.covariates)
    
    def test_getCovariates_mismatch_covs_samples_covariates_dimension(self):
        covs_samples=np.array(('S1', 'S2', 'S3'))
        with self.assertRaises(DataMismatch):
	    self.datainput.getCovariates(covariates=self.covariates,
                                         covs_samples=covs_samples)

    def test_getCovariates_covariates_type(self):
        covariates=((1,2),(2,3))
        with self.assertRaises(TypeError):
	    self.datainput.getCovariates(covariates=covariates,
                                         covs_samples=self.covs_samples)

    def test_getCovariates_covs_samples_type(self):
        covs_samples=['S1', 'S2', 'S3']
        with self.assertRaises(TypeError):
	    self.datainput.getCovariates(covariates=self.covariates,
                                         covs_samples=covs_samples)

    def test_getCovariates_fromOptions_wrongFileFormat(self):
        self.options = Options()
        self.datainputFromOptions = DataInput(self.options)
        with self.assertRaises(FormatError):
	    self.datainputFromOptions.getCovariates()
    
    def test_getCovariates_fromOptions_FileNonExisting(self):
        self.options = Options(file_covariates="test.csv")
        self.datainputFromOptions = DataInput(self.options)
        with self.assertRaises(IOError):
	    self.datainputFromOptions.getCovariates()

    def test_getRelatedness_missing_relatedness_samples(self):
        with self.assertRaises(MissingInput):
	    self.datainput.getRelatedness(relatedness=self.relatedness)
    
    def test_getRelatedness_mismatch_relatedness_and_samples_dimension(self):
        relatedness_samples=np.array(('S1', 'S2', 'S3'))
        with self.assertRaises(DataMismatch):
	    self.datainput.getRelatedness(relatedness=self.relatedness,
                                relatedness_samples=relatedness_samples)
    
    def test_getRelatedness_is_square_matrix(self):
        relatedness = np.array(((1,2),(2,1), (3,3)))
        with self.assertRaises(FormatError):
	    self.datainput.getRelatedness(relatedness=relatedness,
                                relatedness_samples=self.relatedness_samples)

    def test_getRelatedness_relatedness_type(self):
        relatedness=((1,2),(2,3))
        with self.assertRaises(TypeError):
	    self.datainput.getRelatedness(relatedness=relatedness,
                                relatedness_samples=self.relatedness_samples)

    def test_getRelatedness_relatedness_samples_type(self):
        relatedness_samples=['S1', 'S2', 'S3']
        with self.assertRaises(TypeError):
	    self.datainput.getRelatedness(relatedness=self.relatedness,
                                    relatedness_samples=relatedness_samples)
    
    def test_getRelatedness_fromOptions_wrongFileFormat(self):
        self.options = Options()
        self.datainputFromOptions = DataInput(self.options)
        with self.assertRaises(FormatError):
	    self.datainputFromOptions.getCovariates()
    
    def test_getRelatedness_fromOptions_FileNonExisting(self):
        self.options = Options(file_relatedness="test.csv")
        self.datainputFromOptions = DataInput(self.options)
        with self.assertRaises(IOError):
	    self.datainputFromOptions.getRelatedness()

    def test_traitstring_with_wrong_characters(self):
        self.options = Options(traitstring="1.3")
        self.datainputFromOptions = DataInput(self.options)
        with self.assertRaises(FormatError):
            self.datainputFromOptions.subsetTraits()
    
    def test_traitstring_with_traitnumber_gt_number_of_phenotypes(self):
        self.options = Options(traitstring="1,1,5")
        phenotypes = np.array(((1,2,1,3), (1,3,1,3)))
        phenotype_ID = np.array(('ID2','ID2', 'ID3','ID4'))
        pheno_samples = np.array(('S1','S2'))
        self.datainputFromOptions = DataInput(self.options)
        self.datainputFromOptions.getPhenotypes(phenotypes=phenotypes, 
                    pheno_samples=pheno_samples, phenotype_ID=phenotype_ID)
        with self.assertRaises(DataMismatch):
            self.datainputFromOptions.subsetTraits()

    def test_common_samples_no_overlap_pheno_relatedness(self):
        self.datainput.pheno_samples = np.array(('S1','S2'))
        self.datainput.relatedness_samples = np.array(('S3','S4'))
        with self.assertRaises(DataMismatch):
            self.datainput.commonSamples()
    
    def test_common_samples_no_overlap_pheno_relatedness_covs(self):
        self.datainput.pheno_samples = np.array(('S1','S2'))
        self.datainput.relatedness_samples = np.array(('S3','S4'))
        self.datainput.covs_samples = np.array(('S5','S6'))
        with self.assertRaises(DataMismatch):
            self.datainput.commonSamples()

    def test_passing_of_transformation_method(self):
        self.options = Options(transform="coxbox")
        phenotypes = np.array(((1,2,1,3), (1,3,1,3)))
        phenotype_ID = np.array(('ID2','ID2', 'ID3','ID4'))
        pheno_samples = np.array(('S1','S2'))
        covariates = np.array(((1,3), (3,1)))
        covs_samples = np.array(('S1','S2'))
        self.datainputFromOptions = DataInput(self.options)
        self.datainputFromOptions.getPhenotypes(phenotypes=phenotypes, 
                    pheno_samples=pheno_samples, phenotype_ID=phenotype_ID)
        self.datainputFromOptions.getCovariates(covariates=covariates, 
                    covs_samples=covs_samples)
        with self.assertRaises(TypeError):
            self.datainputFromOptions.regress_and_transform()

    def test_regress_phenotypes_and_covs_are_different(self):
        self.options = Options(transform="coxbox", regress=True)
        phenotypes = np.array(((1,2,1,3), (1,3,1,3)))
        phenotype_ID = np.array(('ID2','ID2', 'ID3','ID4'))
        pheno_samples = np.array(('S1','S2'))
        covariates = phenotypes
        covs_samples = np.array(('S1','S2'))
        self.datainputFromOptions = DataInput(self.options)
        self.datainputFromOptions.getPhenotypes(phenotypes=phenotypes, 
                    pheno_samples=pheno_samples, phenotype_ID=phenotype_ID)
        self.datainputFromOptions.getCovariates(covariates=covariates, 
                    covs_samples=covs_samples)
        with self.assertRaises(DataMismatch):
            self.datainputFromOptions.regress_and_transform()
def soon():
    ### make tests for ###

    datalimmbo = DataLimmbo(datainput=datainput, options=dataparse.options)
    resultsQ = datalimmbo.sampleCovarianceMatricesPP()
    datalimmbo.combineBootstrap(resultsQ)

if __name__ == '__main__':
    unittest.main()
