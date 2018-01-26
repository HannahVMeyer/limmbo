import unittest as unittest
import numpy as np
from numpy.random import RandomState
import pandas as pd

from limmbo.core.vdbootstrap import LiMMBo
from limmbo.core.vdbootstrap import DataMismatch
from limmbo.io.input import InputData

class limmbo(unittest.TestCase):

    def setUp(self):
	random = RandomState(1)
	P = 10
	N = 100
	SNP = 1000
	self.data = InputData(verbose=False)
	self.data.phenotypes = random.normal(0,1, (N, P))
	self.data.pheno_samples = np.array(['S{}'.format(x+4) for x in range(N)])
	self.data.phenotype_ID = np.array(['ID{}'.format(x+1) for x in range(P)])
        X = (random.rand(N, SNP) < 0.3).astype(float)
        self.data.relatedness = np.dot(X, X.T)/float(SNP)
        self.relatedness_samples = np.array(['S{}'.format(x+1) for x in range(N)])
        self.limmbo = LiMMBo(datainput=self.data, S=5)

    def test_S_not_greater_than_P(self):
        random = RandomState(1)
        P = 3
        N = 100
        SNP = 1000
        data = InputData(verbose=False)
        data.phenotypes = random.normal(0,1, (N, P))
        data.pheno_samples = np.array(['S{}'.format(x+4) for x in range(N)])
        data.phenotype_ID = np.array(['ID{}'.format(x+1) for x in range(P)])
        X = (random.rand(N, SNP) < 0.3).astype(float)
        data.relatedness = np.dot(X, X.T)/float(SNP)
        relatedness_samples = np.array(['S{}'.format(x+1) for x in range(N)])
        with self.assertRaises(DataMismatch):
            limmbo = LiMMBo(datainput=data, S=5)
    

if __name__ == '__main__':
    unittest.main()
