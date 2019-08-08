import unittest as unittest
import numpy as np
import pandas as pd

from limmbo.io.reader import ReadData
from limmbo.io.reader import MissingInput
from limmbo.io.reader import DataMismatch
from limmbo.io.reader import FormatError


class Reader(unittest.TestCase):

    def setUp(self):
        self.datainput = ReadData(verbose=False)

    def test_getPhenotypes_missing_phenotype_file(self):
        with self.assertRaises(MissingInput):
            self.datainput.getPhenotypes()

    def test_getPhenotypes_wrong_file_format(self):
        with self.assertRaises(FormatError):
            self.datainput.getPhenotypes(file_pheno="pheno.unknownFormat")

    def test_getPhenotypes_file_cannot_be_opened(self):
        with self.assertRaises(IOError):
            self.datainput.getPhenotypes(file_pheno="pheno.csv")

    def test_getCovariates_wrong_file_format(self):
        with self.assertRaises(FormatError):
            self.datainput.getCovariates(file_covariates="covs.unknownFormat")

    def test_getCovariates_file_cannot_be_opened(self):
        with self.assertRaises(IOError):
            self.datainput.getCovariates(file_covariates="covs.csv")

    def test_getGenotypes_missing_file(self):
        with self.assertRaises(MissingInput):
            self.datainput.getGenotypes()

    def test_getGenotypes_wrong_file_format(self):
        with self.assertRaises(FormatError):
            self.datainput.getGenotypes(
                    file_genotypes="genos.unknownFormat")

    def test_getGenotypes_file_cannot_be_opened(self):
        with self.assertRaises(IOError):
            self.datainput.getGenotypes(file_genotypes="genos.csv")

    def test_getPCs_wrong_file_format(self):
        with self.assertRaises(FormatError):
            self.datainput.getPCs(file_pcs="PCAs.unknownFormat")

    def test_getVarianceComponents_file_cannot_be_opened(self):
        with self.assertRaises(IOError):
            self.datainput.getVarianceComponents(file_Cg="cg.csv")

    def test_getVarianceComponents_file_cannot_be_opened(self):
        with self.assertRaises(IOError):
            self.datainput.getVarianceComponents(file_Cg="cg.csv",
                                                 file_Cn="cn_csv")

    def test_traitstring_with_wrong_characters(self):
        with self.assertRaises(FormatError):
            self.datainput.getTraitSubset(traitstring="1.3")


if __name__ == '__main__':
    unittest.main()
