import unittest as unittest
import numpy as np
import pandas as pd

from limmbo.io.input import InputData
from limmbo.io.input import MissingInput
from limmbo.io.input import DataMismatch
from limmbo.io.input import FormatError


class Input(unittest.TestCase):

    def setUp(self):
        self.datainput = InputData()
        self.phenotypes = np.array(((1, 2), (1, 3)))
        self.pheno_samples = np.array(('S1', 'S2'))
        self.phenotype_ID = np.array(('ID1', 'ID2'))

        self.covariates = np.array((1, 2))
        self.covs_samples = np.array(('S1', 'S2'))

        self.relatedness = np.array(((3, 2), (2, 3)))
        self.relatedness_samples = np.array(('S1', 'S2'))

    def test_addPhenotypes_missing_phenotypes(self):
        with self.assertRaises(TypeError):
            self.datainput.addPhenotypes()

    def test_addPhenotypes_mismatch_phenotype_ID_phenotype_dimension(self):
        pheno_ID = np.array(('ID1', 'ID2', 'ID3', 'ID4'))
        with self.assertRaises(DataMismatch):
            self.datainput.addPhenotypes(phenotypes=self.phenotypes,
                                         phenotype_ID=pheno_ID,
                                         pheno_samples=self.pheno_samples)

    def test_addPhenotypes_mismatch_pheno_samples_phenotype_dimension(self):
        pheno_samples = np.array(('S1', 'S2', 'S3'))
        with self.assertRaises(DataMismatch):
            self.datainput.addPhenotypes(phenotypes=self.phenotypes,
                                         phenotype_ID=self.phenotype_ID,
                                         pheno_samples=pheno_samples)

    def test_addCovariates_mismatch_covs_samples_covariates_dimension(self):
        covs_samples = np.array(('S1', 'S2', 'S3'))
        with self.assertRaises(DataMismatch):
            self.datainput.addCovariates(covariates=self.covariates,
                                         covs_samples=covs_samples)

    def test_addRelatedness_mismatch_relatedness_and_samples_dimension(self):
        relatedness_samples = np.array(('S1', 'S2', 'S3'))
        with self.assertRaises(DataMismatch):
            self.datainput.addRelatedness(
                relatedness=self.relatedness,
                relatedness_samples=relatedness_samples)

    def test_addRelatedness_is_square(self):
        relatedness = np.array(((1, 2), (2, 1), (3, 3)))
        with self.assertRaises(FormatError):
            self.datainput.addRelatedness(
                relatedness=relatedness,
                relatedness_samples=self.relatedness_samples)

    def test_addRelatedness_is_symmetric(self):
        relatedness = np.array(((1, 2), (6, 1)))
        with self.assertRaises(FormatError):
            self.datainput.addRelatedness(
                relatedness=relatedness,
                relatedness_samples=self.relatedness_samples)

    def test_addRelatedness_is_positive_semidefinite(self):
        relatedness = np.array(((1, 2), (2, 1)))
        with self.assertRaises(FormatError):
            self.datainput.addRelatedness(
                relatedness=relatedness,
                relatedness_samples=self.relatedness_samples)

    def test_traitstring_with_traitnumber_gt_number_of_phenotypes(self):
        phenotypes = np.array(((1, 2, 1, 3), (1, 3, 1, 3)))
        phenotype_ID = np.array(('ID1', 'ID2', 'ID3', 'ID4'))
        pheno_samples = np.array(('S1', 'S2'))
        self.datainput = InputData(verbose=False)
        self.datainput.addPhenotypes(phenotypes=phenotypes,
                                     pheno_samples=pheno_samples,
                                     phenotype_ID=phenotype_ID)
        with self.assertRaises(DataMismatch):
            self.datainput.subsetTraits(traitlist=[1, 2, 5])

    def test_common_samples_no_overlap_pheno_relatedness(self):
        pheno_samples = np.array(('S10', 'S30'))
        self.datainput = InputData(verbose=False)
        self.datainput.addPhenotypes(phenotypes=self.phenotypes,
                                     pheno_samples=pheno_samples,
                                     phenotype_ID=self.phenotype_ID)
        self.datainput.addRelatedness(
            relatedness=self.relatedness,
            relatedness_samples=self.relatedness_samples)
        with self.assertRaises(DataMismatch):
            self.datainput.commonSamples()

    def test_common_samples_no_overlap_pheno_relatedness_covs(self):
        pheno_samples = np.array(('S10', 'S20'))
        self.datainput = InputData(verbose=False)
        self.datainput.addPhenotypes(phenotypes=self.phenotypes,
                                     pheno_samples=pheno_samples,
                                     phenotype_ID=self.phenotype_ID)
        self.datainput.addCovariates(covariates=self.covariates,
                                     covs_samples=self.covs_samples)
        self.datainput.addRelatedness(
            relatedness=self.relatedness,
            relatedness_samples=self.relatedness_samples)
        with self.assertRaises(DataMismatch):
            self.datainput.commonSamples()

    def test_common_samples_no_overlap_samplelist(self):
        self.datainput = InputData(verbose=False)
        self.datainput.addPhenotypes(phenotypes=self.phenotypes,
                                     pheno_samples=self.pheno_samples,
                                     phenotype_ID=self.phenotype_ID)
        self.datainput.addRelatedness(
            relatedness=self.relatedness,
            relatedness_samples=self.relatedness_samples)
        with self.assertRaises(DataMismatch):
            self.datainput.commonSamples(samplelist=["S3", "S4"])

    def test_common_samples_samplelist_longer_than_common_samples(self):
        self.datainput = InputData(verbose=False)
        self.datainput.addPhenotypes(phenotypes=self.phenotypes,
                                     pheno_samples=self.pheno_samples,
                                     phenotype_ID=self.phenotype_ID)
        self.datainput.addRelatedness(
            relatedness=self.relatedness,
            relatedness_samples=self.relatedness_samples)
        with self.assertRaises(DataMismatch):
            self.datainput.commonSamples(samplelist=["S1", "S2", "S3", "S4"])

    def test_passing_of_transformation_method(self):
        phenotypes = np.array(((1, 2, 1, 3), (1, 3, 1, 3)))
        phenotype_ID = np.array(('ID1', 'ID2', 'ID3', 'ID4'))
        pheno_samples = np.array(('S1', 'S2'))
        covariates = np.array(((1, 3), (3, 1)))
        covs_samples = np.array(('S1', 'S2'))
        self.datainputFromOptions = InputData(verbose=False)
        self.datainputFromOptions.addPhenotypes(phenotypes=phenotypes,
                                                pheno_samples=pheno_samples,
                                                phenotype_ID=phenotype_ID)
        self.datainputFromOptions.addCovariates(covariates=covariates,
                                                covs_samples=covs_samples)
        with self.assertRaises(TypeError):
            self.datainputFromOptions.transform(transform="coxbox")

    def test_regress_phenotypes_and_covs_are_different(self):
        phenotypes = np.array(((1, 2, 1, 3), (1, 3, 1, 3)))
        phenotype_ID = np.array(('ID1', 'ID2', 'ID3', 'ID4'))
        pheno_samples = np.array(('S1', 'S2'))
        covariates = phenotypes
        covs_samples = np.array(('S1', 'S2'))
        self.datainputFromOptions = InputData(verbose=False)
        self.datainputFromOptions.addPhenotypes(phenotypes=phenotypes,
                                                pheno_samples=pheno_samples,
                                                phenotype_ID=phenotype_ID)
        self.datainputFromOptions.addCovariates(covariates=covariates,
                                                covs_samples=covs_samples)
        with self.assertRaises(DataMismatch):
            self.datainputFromOptions.regress()

        self.datainputFromOptions.addCovariates(covariates=self.covariates,
                                                covs_samples=self.covs_samples)

    def test_transform_method_is_supported(self):
        self.datainput.addPhenotypes(phenotypes=self.phenotypes,
                                     pheno_samples=self.pheno_samples,
                                     phenotype_ID=self.phenotype_ID)
        with self.assertRaises(TypeError):
            self.datainput.transform(type="blub")

    def test_addVarianceComponents_Cg_is_square(self):
        Cg = np.array(((1, 2), (2, 1), (3, 3)))
        Cn = np.array(((1, 2), (2, 1)))
        with self.assertRaises(FormatError):
            self.datainput.addVarianceComponents(Cg=Cg, Cn=Cn)

    def test_addVarianceComponents_Cn_is_symmetric(self):
        Cg = np.array(((1, 2), (2, 5)))
        Cn = np.array(((1, 2), (2, 1)))
        with self.assertRaises(FormatError):
            self.datainput.addVarianceComponents(Cg=Cg, Cn=Cn)

    def test_addVarianceComponents_Cg_is_positive_semidefinite(self):
        Cg = np.array(((1, 2), (2, 1)))
        Cn = np.array(((3, 2), (2, 3)))
        with self.assertRaises(FormatError):
            self.datainput.addVarianceComponents(Cg=Cg, Cn=Cn)

    def test_addVarianceComponents_Cn_is_square(self):
        Cn = np.array(((1, 2), (2, 1), (3, 3)))
        Cg = np.array(((1, 2), (2, 1)))
        with self.assertRaises(FormatError):
            self.datainput.addVarianceComponents(Cg=Cg, Cn=Cn)

    def test_addVarianceComponents_Cn_is_symmetric(self):
        Cn = np.array(((1, 2), (3, 1)))
        Cg = np.array(((1, 2), (2, 1)))
        with self.assertRaises(FormatError):
            self.datainput.addVarianceComponents(Cg=Cg, Cn=Cn)

    def test_addVarianceComponents_Cn_is_positive_semidefinite(self):
        Cn = np.array(((1, 2), (2, 1)))
        Cg = np.array(((3, 2), (2, 3)))
        with self.assertRaises(FormatError):
            self.datainput.addVarianceComponents(Cg=Cg, Cn=Cn)

if __name__ == '__main__':
    unittest.main()
