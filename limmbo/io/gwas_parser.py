

###############
### modules ###
###############


import sys

# plotting
import argparse
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from matplotlib.backends.backend_pdf import PdfPages

# stats module
import GPy
import scipy as sp
import scipy.linalg as la
import scipy.stats as st

import pandas as pd
import numpy as np
import pylab as pl
import h5py



# import LIMIX tools
import limix
import limix.modules.varianceDecomposition as var
import limix.modules.qtl as qtl
import limix.io.data as data
import limix.io.genotype_reader as gr
import limix.io.phenotype_reader as phr
import limix.io.data_util as data_util
import limix.utils.preprocess as preprocess
import limix.utils.plot as plot
from limix.stats.geno_summary import *

import mtSet.pycore.modules.multiTraitSetTest as MTST
from mtSet.pycore.utils.normalization import gaussianize
from mtSet.pycore.utils.utils import smartAppend
from mtSet.pycore.utils.utils import smartDumpDictHdf5
from mtSet.pycore.utils.normalization import regressOut
from mtSet.pycore.utils.fit_utils import fitPairwiseModel
from mtSet.pycore.utils.fit_utils import fitSingleTraitModel

# other requirements
import os
import cPickle
from math import sqrt
import copy
import re
from distutils.util import strtobool
import pdb

#######################
### input functions ###
#######################

class DataParse(object):
    def __init__(self):
        '''
        nothing to initialize
        '''
        self.options = None

    def getArgs(self, debug=False):
        parser = argparse.ArgumentParser(description="Python2.7 script. Use LIMIX and mtSet modules for GWAS.")
        parser.add_argument('-pf', '--file_pheno', action="store", 
                dest="file_pheno", required=True, 
                help='HF5 phenotype file (via limix_format_data.py) or [N x P] .csv file (first column: sample IDs, first row: phenotype IDs)')
        parser.add_argument('-gf', '--file_geno', action="store", 
                dest="file_geno", required=False, default=None, 
                help='HF5 genotype file (via limix_format_data.py) or [S x N].csv file (first column: SNP id, first row: sample IDs).')
        parser.add_argument('-cf', '--file_covariates',  action="store", 
                dest="file_covariates", required=False, default = None, 
                help="Path to [N x C] file of covariates matrix (first column: sample IDs, first row: phenotype IDs)")
        parser.add_argument('-kf', '--file_kinship',  action="store", 
                dest="file_kinship", required=False, default = None, 
                help="Path to [N x N] file of kinship matrix for linear mixed model (first rows: sample IDs)")
        parser.add_argument('-pcsf', '--file_pcs', action="store", 
                dest="file_pcs", required=False, default=None, 
                help="Path to [N x PCs] file of principal components from genotypes to be included as covariates (first column: sample IDs, first row: PC IDs); Default: None")
        parser.add_argument('-cgf', '--file_cg', action="store", 
                dest="file_Cg", required=False, default=None, 
                help="Only needed in lmm multitrait analysis/setup: input/output file name for genetic covariance matrix (rows: traits, columns: traits)")
        parser.add_argument('-cnf', '--file_cn', action="store", 
                dest="file_Cn", required=False, default=None, 
                help="Only needed in lmm multitrait analysis/setup: input/output file name for noise covariance matrix (rows: traits, columns: traits)")
        parser.add_argument('-sf', '--file_samplelist', action="store", 
                dest="file_samplelist", required=False, default=None, 
                help="Path to file for sample list used for sample filtering from phenotypes and genotypes")

        parser.add_argument('-c', '--chromosome', action="store", 
                dest="chromosome", required=False, help='Chromosome')
        parser.add_argument('-seed', '--seed', action="store", 
                dest="seed", required=False, default=474,
                help='Seed for permutation of genotypes. Default: 474', 
                type=int)
        parser.add_argument('-nosamplematch', '--nosamplematch',
                action="store_true", dest="nosamplematch", required=False, 
                default=False, help="") 
        parser.add_argument('-noPlot', '--noPlot', action="store_true", 
                dest="noPlot", required=False, default=False, 
                help='Should GWAS results be plotted; Default: True')
        parser.add_argument('-noCache', '--noCache', action="store_false", 
                dest="cache", required=False, default=True, 
                help='Should mtSet be cached; Default: True')

        # settings: data transform/subset options
        parser.add_argument('-tr', '--transform_method', action="store", 
                dest="transform", default= 'scale', required=False, 
                type=str, 
                help='Type of data preprocessing: scale or gaussianize')
        parser.add_argument('-traitset', '--traitset', action="store", 
                dest="traitstring", required=False,default=None,
                help='which traits to choose; default: None (=all traits)')
        parser.add_argument('-sampleset', '--sampleset', action="store", 
                dest="samplestring", required=False, default=None,
                help='which samples to choose; default: None (=all traits)')
        parser.add_argument('-nrpcs', '--nrpcs', action="store", 
                dest="nrpcs", required=False,default=10,
                help='First PCs to chose; default: 10', type= int)
        parser.add_argument('-reg', '--reg_covariates', 
                action="store_true", dest="regress", required=False,  
                help='Should covariates be regressed out? Default: False')
        parser.add_argument('-standardise', '--standardise', 
                action="store_true", dest="standardise", required=False,  
                help='Should genotypes be standardised? Default: False')
        
        # output settings
        parser.add_argument('-of', '--output', action="store", 
                dest="output", required=False, help='Output filename')
        parser.add_argument('-fileend', '--fileend', action="store", 
                dest="fileend", required=False,  default="",
                help='string to be appened to all output files')
        parser.add_argument('-v', '--verbose', action="store_true", 
                dest="verbose", required=False, default=False, 
                help="Should analysis step description be printed; default:False")

        # analysis setup
        parser.add_argument('-m', '--mode', action="store", dest="mode", 
                required=True, 
                help='Mode for running the analysis:')
        parser.add_argument('-set', '--set-up', action="store", 
                dest="setup", required=False,default='lmm', 
                help='Test set up: lm or lmm')
        parser.add_argument('-searchDelta', '--searchDelta', 
                action="store_true", dest="searchDelta", required=False, 
                default=False,  
                help='SearchDelta for lmm testing? Default: False')
        parser.add_argument('-freqonly', '--freqonly', action="store_true", 
                dest="freqonly", required=False, default=False,  
                help='Only get allele frequencies? Default: False')

        # significance settings	    
        parser.add_argument('-permute', '--permute', action="store_true", 
                dest="permute", required=False, default=False,  
                help='Permute genotypes? Default: False')
        parser.add_argument('-empiricalP', '--empiricalP', 
                action="store_true", dest="empiricalP", required=False, 
                default=False,  
                help='Compute empirical pvalues? Default: False')
        parser.add_argument('-computeFDR', '--computeFDR', 
                action="store_true", dest="computeFDR", required=False, 
                default=False,  
                help='Compute false discovery rate? Default: False')
        parser.add_argument('-fdr', '--fdr', action="store", dest="fdr", 
                required=False, default=0.01,  
                help='FDR threshold? Default: 0.01', type=float)
        parser.add_argument('-likelihoods', '--likelihoods', 
                action="store_true",  dest="likelihoods", required=False, 
                default=False, help='Return likelihoods. Default: False')
        parser.add_argument('-meff', '--meff', action="store", dest="meff", 
                required=False, default=None,  
                help="singletrait mode: effective number of tests (when testing multiple traits) to be corrected for;  Default: number of traits", type=float)
	
	self.options = parser.parse_args()

	if self.options.file_geno is None and \
                self.options.file_kinship is None:
            parser.error("At least one of -gf and --kf required")
	if self.options.setup == 'lmm' and self.options.file_kinship is None:
            parser.error("For -set lmm, --kf is required")
	if self.options.file_covariates is None and \
                self.options.file_pcs is None and self.options.regress is True:
            parser.error(("Regress is set to True but neither covariate file",
            "nor PC file provided"))

	return self

class DataInput(object):
    """
    """
    def __init__(self, options=None):
        '''
        nothing to initialize
        '''
        self.options = options
	self.samples = None
	self.phenotypes = None
	self.pheno_samples = None
	self.phenotype_ID = None
	self.snps = None
	self.position = None
	self.geno_samples = None
	self.covariates = None
	self.covariate_samples = None
	self.relatedness = None
	self.relatedness_samples = None
	self.pcs = None
	self.pc_samples = None
	self.Cg = None
	self.Cn = None
        self.trainset = None
        self.traitsarray = None

    def getPhenotypes(self): 
	verboseprint("Extracting phenotypes", verbose=self.options.verbose)
	if re.search(".h5", self.options.file_pheno) is None:
	    self.phenotypes = pd.io.parsers.read_csv(self.options.file_pheno,
                    index_col=0)
	    self.phenotype_ID = np.array(self.phenotypes.columns)
	    self.pheno_samples = np.array(self.phenotypes.index)
	    self.phenotypes = np.array(self.phenotypes)
	else:
	    file  = h5py.File(self.options.file_pheno,'r')
	    self.phenotypes = file['phenotype']['matrix'][:]
	    self.phenotype_ID = np.array(file['phenotype']['col_header']\
                    ['phenotype_ID'][:].astype('str'))
	    self.pheno_samples = np.array(file['phenotype']['row_header']\
                    ['sample_ID'][:].astype('str'))
	    self.phenotypes = np.array(self.phenotypes)
	    file.close()
	return self
    
    def getGenotypes(self): 
       
        if re.search(".h5", self.options.file_geno) is None:
	    verboseprint("Extracting genotypes from .csv file", 
                    verbose=self.options.verbose)
            genotypes = pd.io.parsers.read_csv(self.options.file_geno, 
                    index_col=0, header=0)
	    snp_info = np.array(genotypes.index)
           
            position = []
            snp_ID = []
            for id in range(snp_info.shape[0]):
                split = np.array(snp_info[id].split('-'))
                snp_ID.append(split[2])
                position.append(split[[0,1]])

	    self.geno_samples = np.array(genotypes.columns)
	    self.snps = np.array(genotypes).astype(float).T
            self.position = pd.DataFrame(np.array(position), 
                    columns=['chrom', 'pos'], index=snp_ID)
        else:
            # read genotype information: chromosome-wise hf5 files
            geno_reader  = gr.genotype_reader_h5py(self.options.file_geno)
            verboseprint("Extracting genotypes from hf5 file", 
                    verbose=self.options.verbose)
            self.geno_samples = geno_reader.sample_ID
            self.snps = geno_reader.getGenotypes().astype(float)
	    self.position = geno_reader.getPos()
        
        if self.options.standardise:
	    verboseprint("Standardise genotypes", verbose=self.options.verbose)
            self.standardiseGenotypes()
	return self

    def getCovariates(self):
	if self.options.file_covariates != None:
	    verboseprint("Reading covariates file", 
                    verbose=self.options.verbose)
	    self.covariates = pd.io.parsers.read_csv(
                    self.options.file_covariates)
	    self.covs_samples = np.ravel(self.covariates.iloc[:,:1])
	    self.covariates = np.array(self.covariates.iloc[:, 1:]).astype(
                    float)
	    # Bug in LIMIX: this concatenation should be done in QTL function; 
            # adjusts for mean of covariates
	    if not self.options.regress:
		self.covariates = sp.concatenate([self.covariates, 
                    sp.ones((self.covariates.shape[0],1))],1)
		self.covariates = np.array(self.covariates)
	else:
            # When cov are None, LIMIX internally sets: covs=sp.ones((N, 1))
	    verboseprint("No covariates set", verbose=self.options.verbose)
	    self.covariates = None
	    self.covs_samples = None
	return self

    def getKinship(self):
	if self.options.file_kinship != None:
	    verboseprint("Reading relationship matrix", 
                    verbose=self.options.verbose)
	    self.relatedness = pd.io.parsers.read_csv(
                    self.options.file_kinship)
	    self.relatedness_samples = np.array(self.relatedness.columns)
	    self.relatedness = np.array(self.relatedness).astype(float)
	else:
	    self.relatedness=None
            verboseprint("No relationship matrix set", 
                    verbose=self.options.verbose)
	    self.relatedness_samples = None
    	return self

    def getPCs(self):
	if self.options.file_pcs != None:
	    verboseprint("Reading PCs", verbose=self.options.verbose)
	    self.pcs = pd.io.parsers.read_csv(self.options.file_pcs, 
                    header=None,sep=" ")
	    self.pc_samples = np.array(self.pcs.iloc[:,:1]).flatten()
	    self.pcs = np.array(self.pcs.iloc[:,2:]).astype(float)
	    verboseprint("Extracting first %s pcs" % self.options.nrpcs, 
                    verbose=self.options.verbose)
	    self.pcs = self.pcs[:, :self.options.nrpcs] 
	else:
	    self.pcs=None
	    verboseprint("No pcs set", verbose=self.options.verbose)
	    self.pc_samples = None
	return self

    def getVD(self):
    	if self.options.mode == 'multitrait':
	    if self.options.file_Cg is None and self.options.file_Cn is None:
		verboseprint(("No variance components supplied, run VD/limmbo",
                "before lmm test"), verbose=self.options.verbose)
		self.Cg, self.Cn = None, None
	    elif self.options.file_Cg is None or self.options.file_Cn is None:
		verboseprint(("Both variant components need to be supplied:",
                "Cg is %s and Cn is %s") % (self.options.file_Cg, 
                    self.options.file_Cn), verbose=self.options.verbose)
		self.Cg, self.Cn = None, None
	    else:
		self.Cg = np.array(pd.io.parsers.read_csv(
                    self.options.file_Cg, header=None))
		self.Cn = np.array(pd.io.parsers.read_csv(
                    self.options.file_Cn, header=None))
	    return self

    def standardiseGenotypes(self):
        for snp in range(self.snps.shape[1]):
            p, q = AlleleFrequencies(self.snps[:, snp])
            var_snp = sqrt(2*p*q)
            for n in range(self.snps[:,snp].shape[0]):
                self.snps[n, snp] = (self.snps[n, snp] - 2*q)/var_snp
        return self
    
    def getAlleleFrequencies(self):
        verboseprint("Get allele frequencies of %s snps from chromosome %s" %
                (self.snps.shape[1], self.options.chromosome))
        self.freqs = np.zeros((self.snps.shape[1],3))
        for snp in range(self.snps.shape[1]):
            self.freqs[snp,1], self.freqs[snp,2] = AlleleFrequencies(
                    self.snps[:, snp])
        self.freqs = self.freqs.astype('str')
        self.freqs[:,0] = np.array(self.position.index)

        pd.DataFrame(self.freqs, columns=["SNP_ID", "A1", "A2"]).to_csv(
                "%s/allelefrequencies_%s.csv" % (self.options.output, 
                    self.options.chromosome), index=False, header=True)
        return self


    def subsetTraits(self):
	if self.options.traitstring is not None:
	    verboseprint("Get subset of %s traits" % self.options.traitstring, 
                    verbose=self.options.verbose)
	    traitslist = [x.split('-') \
                    for x in self.options.traitstring.split(',')]
	    self.traitsarray = []
	    for t in traitslist:
		if len(t) == 1:
		   self.traitsarray.append(int(t[0]) -1)
		else:
		   [self.traitsarray.append(x) \
                           for x in range(int(t[0]) -1, int(t[1]))]
		
	    self.phenotypes = self.phenotypes[:,self.traitsarray]
	    self.phenotype_ID = self.phenotype_ID[self.traitsarray]
	    if self.Cg is not None and \
                    self.phenotypes.shape[1] != self.Cg.shape[1] :
		self.Cg = self.Cg[self.traitsarray,:][:, self.traitsarray]
		self.Cn = self.Cn[self.traitsarray,:][:, self.traitsarray]
	return self
    
    def commonSamples(self):
        if self.options.file_geno is not None:
            if self.options.file_samplelist is not None or \
                self.options.samplestring is not None:
                
                verboseprint("Read sample list to be extracted from " \
                "phenotypes (samples:%s) and genotypes (samples: %s)" % 
                (len(self.pheno_samples), len(self.geno_samples)), 
                verbose=self.options.verbose)

                if self.options.file_samplelist is not None:
                    # read sample list
                    subset = np.array(pd.io.parsers.read_csv(
                        self.options.file_samplelist, header=None))
                else:
	            subset = np.array(self.options.samplestring.split(","))
        
                pdb.set_trace()
                verboseprint("Number of samples in sample list: %s" % 
                    len(subset))
            else:
	        verboseprint("Get common samples between phenotypes " \
                        "(samples:%s) and genotypes (samples: %s)" % \
                        (len(self.pheno_samples), len(self.geno_samples)), 
                verbose=self.options.verbose)
	        # get common samples between genotypes and phenotypes
	        subset = np.intersect1d(self.pheno_samples, self.geno_samples)
	        verboseprint("Number of common samples between phenotypes " \
                        "and genotypes: %s" % len(subset), 
                        verbose=self.options.verbose)
            
            # subsample arrays and match order of phenotypes/covariates/kinship 
            # and respective samples to genotypes
            verboseprint("Match order of pheno_samples to geno_samples and" \
            " extract corresponding samples in right order from additional," \
            " optional files (kinship, covariates, pcs)", 
            verbose=self.options.verbose)		
            subset_geno_samples = np.in1d(self.geno_samples, subset)
            self.geno_samples = self.geno_samples[subset_geno_samples]
            self.snps = self.snps[subset_geno_samples,:]
            self.samples = self.geno_samples
            if self.options.permute is True:
                verboseprint("Permuting genotype samples (seed %s)" % 
                    self.options.seed, verbose=self.options.verbose)
                self.snps = self.snps[np.random.RandomState(
                    self.options.seed).choice(self.snps.shape[0], 
                        self.snps.shape[0], replace=False), :]
	
            subset_pheno_samples = np.in1d(self.pheno_samples, subset)
            self.pheno_samples = self.pheno_samples[subset_pheno_samples]
            self.phenotypes = self.phenotypes[subset_pheno_samples, :] 
            self.phenotypes, self.pheno_samples, samples_before, \
                    samples_after = match(self.geno_samples, 
                        self.pheno_samples, self.phenotypes, 
                        squarematrix=False)
        else:
            subset = self.pheno_samples
            self.samples = self.pheno_samples
	
	if self.relatedness is not None:
            subset_relatedness_samples = np.in1d(self.relatedness_samples, 
                    subset)
            self.relatedness_samples = self.relatedness_samples[\
                subset_relatedness_samples]
            self.relatedness = self.relatedness[subset_relatedness_samples, :] 
            self.relatedness, self.relatedness_samples, samples_before,\
                    samples_after = match(self.samples, 
                        self.relatedness_samples, 
                        self.relatedness, squarematrix=True)
	
	if self.covariates is not None:
	    subset_covs_samples = np.in1d(self.covs_samples, subset)
	    self.covs_samples = self.covs_samples[subset_covs_samples]
	    self.covariatess = self.covariates[subset_covs_samples, :] 
	    self.covariates, self.covs_samples, samples_before, samples_after \
                    = match(self.samples, self.covs_samples, self.covariates, 
                            squarematrix=False)
	
	if self.pcs is not None:
	    subset_pc_samples = np.in1d(self.pc_samples, subset)
	    self.pc_samples = self.pc_samples[subset_pc_samples]
	    self.pcs = self.pcs[subset_pc_samples, :] 
	    self.pcs, self.pc_samples, samples_before, samples_after = \
                    match(self.samples, self.pc_samples, self.pcs, 
                            squarematrix=False)
    
    def regress_and_transform(self):
	if self.options.regress:
            type = "covariates"
	    if self.pcs is not None:
                if self.covariates is not None:
	    	    verboseprint("Append Pcs to covariates", 
                            verbose=self.options.verbose)
		    self.covariates = sp.concatenate([self.covariates, 
                        self.pcs],axis=1)
                    type = "covariates and PCs"
                else:
		    self.covariates = self.pcs
                    type = "PCs"
	    verboseprint("Regress out %s" % type, verbose=self.options.verbose)
	    self.phenotypes = regressOut(self.phenotypes, self.covariates)
	    self.covariates = None
            self.transform()
	return self
    
    def transform(self):
        if self.options.transform == "scale":
            verboseprint("Use %s as transformation" % self.options.transform, 
                    verbose=self.options.verbose)
            self.phenotypes = scale(self.phenotypes)
        elif self.options.transform == "gaussian":
            verboseprint("Use %s as transformation" % self.options.transform, 
                    verbose=self.options.verbose)
            self.phenotypes = gaussianize(self.phenotypes)
        else:
            verboseprint("Data is not transformed", 
                    verbose=self.options.verbose)
        return self
	

class DataGWAS(object):
    """
    """
    def __init__(self, datainput, options=None):
        '''
        nothing to initialize
        '''
	self.options = options
	self.snps = datainput.snps
	self.position = datainput.position
        self.phenotypes = datainput.phenotypes
        self.phenotype_ID = datainput.phenotype_ID
        self.covariates = datainput.covariates
        self.relatedness = datainput.relatedness
        self.traitsarray = datainput.traitsarray
        self.pcs = datainput.pcs
	self.Cg = datainput.Cg
	self.Cn = datainput.Cn
	self.pvalues = None
	self.pvalues_adjust = None
        self.pvalues_empirical_raw = None
        self.betas = None
        self.stats = None
        self.z = None
    
    
    ############################
    ### core functions GWAS: ###
    ############################
    
    def run_GWAS(self):

	# getting SNP info
	verboseprint("extracting SNP info")
	SNP = np.array(self.position.index)
	CHR = np.array(self.position.iloc[:,:1])
	POS = np.array(self.position.iloc[:,1:])

	# set parameters for the analysis
	N, P = self.phenotypes.shape
	S    = self.snps.shape[1]
	verboseprint("loaded %d samples, %d phenotypes, %s snps" % (N,P,S))
	verboseprint("Set searchDelta %s" % self.options.searchDelta)
	test = "lrt"
        self.pvalues_empirical=None

	if self.options.mode == "multitrait":
	    self.pvalues, self.betas, self.NLL0, self.NLLAlt,  model = \
                    self.run_mt_anyeffect_GWAS(N, P, S, test)
	    self.stats = None
	    self.z = None

        if self.options.mode == "singletrait":
	    self.pvalues, self.pvalues_adjust, self.betas, self.stats, \
                    self.z, model = \
                    self.run_st_GWAS(N, P, S, test)
	
        if self.options.empiricalP:
            self.computeEmpiricalP(N, P, S, test)
            if self.options.noPlot is False:
	        self.manhattanQQ(model=model, P=P, empiricalP=True)
        elif self.options.computeFDR:
            self.computeFDR(N, P, S, model, test)

        if self.options.permute:
	    model = "%s_permute%s" % (model, self.options.seed)
	
        pvalues_out = self.writeResult(model=model, CHR=CHR, SNP=SNP, POS=POS)
        if self.options.noPlot is False:
            self.manhattanQQ(model=model, P=P)

    def run_mt_anyeffect_GWAS(self, N, P, S, test, empiricalP=False, 
            computeFDR=False):
	if self.covariates is None:
	    Acovs = None
	else:
            Acovs = sp.eye(P)
	if self.options.setup == "lmm":
	    if self.Cg is None:
                if P > 30:
                    print("Warning: With more than 30 traits, VD is unlikely" \
                    " to work, consider bootstrapping trait-trait covariance" \
                    " components")
                verboseprint("Estimate Variance components")
		self.Cg, self.Cn = self.varianceDecomposition(N=N,P=P)
	    K1c = self.Cg
	    K2c = self.Cn
	    K1r = self.relatedness
	    model="lmm_mt"
	else:
	    K1c = 1e-9*sp.eye(P)
	    K2c = sp.cov(self.phenotypes.T)
	    K1r = sp.eye(N)
            
	    if self.pcs is not None:
		model="lm_mt_pcs"
	    else:
		model="lm_mt"
	Asnps = sp.eye(P)
        if empiricalP or computeFDR:
	    lm, pvalues = qtl.test_lmm_kronecker(snps=self.snps_permute, 
                    phenos=self.phenotypes, Asnps=Asnps, Acovs=Acovs, 
                    covs=self.covariates, K1r=K1r, K1c=K1c, K2c=K2c, 
                    searchDelta=self.options.searchDelta)
            betas, NLL0, NLLAlt = np.array([[], [], []])
        else:
            verboseprint("Computing multi-trait (any effect) model: %s" 
                    % model)
            lm, pvalues = qtl.test_lmm_kronecker(snps=self.snps, 
                    phenos=self.phenotypes, Asnps=Asnps, Acovs=Acovs, 
                    covs=self.covariates, K1r=K1r, K1c=K1c, K2c=K2c, 
                    searchDelta=self.options.searchDelta)
	    betas = lm.getBetaSNP()
            NLL0 = lm.getNLL0()
            NLLAlt = lm.getNLLAlt()
	return pvalues, betas, NLL0, NLLAlt, model	    


    def run_st_GWAS(self, N, P, S, test, empiricalP=False, computeFDR=False):
	
        if self.options.setup == "lmm":
	    model="lmm_st"
	    K = self.relatedness
	else:
            if self.pcs is not None:
		model="lm_st_pcs"
	    else:
		model="lm_st"
	    K = None
	    
        if empiricalP or computeFDR:
	    lm = qtl.test_lmm(snps=self.snps_permute, pheno=self.phenotypes, 
                    K=K, covs=self.covariates, test=test)
	    pvalues = lm.getPv()
            betas, stats, z, NLL0, NLLAlt = np.array([[], [], [], [], []])
	else:
	    verboseprint("Computing single-trait linear model (%s)" % model)
            lm = qtl.test_lmm(snps=self.snps, pheno=self.phenotypes, K=K, 
                    covs=self.covariates, test=test)
	    verboseprint("Extracting p and beta values")

	    pvalues = lm.getPv()
	    betas = lm.getBetaSNP()
	    stats = lm.test_statistics
            #NLL0 = lm.NLL_0
            #NLLAlt = lm.NLL_alt
	    z = sp.sign(betas) * sp.sqrt(st.chi2(1).isf(pvalues))
	
        if self.options.meff is None:
            self.options.meff = P
	    
        pvalues_adjust = np.array([adjustForMeff(p, self.options.meff) \
                for p in pvalues])

	return pvalues, pvalues_adjust, betas, stats, z, model

    def varianceDecomposition(self, N, P, method='mtSet', cache=True):
	verboseprint("Method is %s" %  method)
	if method is 'limix':
	    Asnps = sp.eye(N)
	    vc = var.VarianceDecomposition(self.phenotypes)
	    vc.addRandomEffect(K=self.relatedness,trait_covar_type='freeform')
	    vc.addRandomEffect(is_noise=True,trait_covar_type='freeform')
	    opt_success = vc.optimize(init_method='random', n_times=30)

	    Cg = vc.getTraitCovar(0)
	    Cn = vc.getTraitCovar(1)

	if method is 'mtSet':
	    out_file = "%s/mtSetresults_nrsamples%s_nrtraits%s.h5" % \
            (self.options.output, N, P)
	    mtSet    = MTST.MultiTraitSetTest(Y=self.phenotypes, 
                    XX=self.relatedness)
	    mtSet_null_info = mtSet.fitNull(cache=self.options.cache, 
                    fname=out_file, n_times=1000, rewrite=True)
	    if mtSet_null_info['conv']:
		verboseprint("mtSet converged")
		Cg = mtSet_null_info['Cg'] + 1e-4*sp.eye(P)
		Cn = mtSet_null_info['Cn'] + 1e-4*sp.eye(P)
                time = mtSet_null_info['time']
	    else:
		sys.exit("mtSet did not converge")

	# write Cg and Cn
	pd.DataFrame([time]).to_csv("%s/timeVarianceDecomposition_%s.csv" % 
                (self.options.output, method), index=False, header=False)
	pd.DataFrame(Cg).to_csv("%s/Cg_%s.csv" % (self.options.output, method), 
                index=False, header=False)
	pd.DataFrame(Cn).to_csv("%s/Cn_%s.csv" % (self.options.output, method), 
                index=False, header=False)
	return(Cg, Cn)

    def computeEmpiricalP(self, N, P, S, test):
        verboseprint("Computing permutation")
        np.random.seed(seed=self.options.seed)
        tests = int(1/self.options.fdr)
        
        if self.options.mode == "multitrait":
            self.count = np.zeros((tests, S))
        else:
            self.pvalues_adjust_min = self.pvalues_adjust.min(axis=0)
            self.count_raw = np.zeros((tests, P, S))
            self.count = np.zeros((tests, S))

        for ps in range (tests):
            verboseprint("Permutation %s" % ps)
            self.snps_permute = self.snps[np.random.choice(N, N, 
                replace=False), :]
            if self.options.mode == "multitrait":
                pvalues_permute, betas_permute, NLL0, NLLAlt, model = \
                        self.run_mt_anyeffect_GWAS(N, P, S, test, 
                                empiricalP=True)
                self.count[ps, :] = pvalues_permute <= self.pvalues
            if self.options.mode == "singletrait":
                pvalues_permute, pvalues_permute_adjust, betas_permute, \
                stats_permute, z_permute, model = self.run_st_GWAS(N, P, S, 
                        test, empiricalP=True)
                self.count_raw[ps,:, :] = pvalues_permute <= self.pvalues
                self.count[ps, :] = pvalues_permute_adjust.min(axis=0) <= \
                    self.pvalues_adjust_min
                self.pvalues_empirical_raw = np.sum(self.count_raw, 
                        axis=0)/tests

        self.pvalues_empirical = np.sum(self.count, axis=0)/tests
        return(self)

    def computeFDR(self, N, P, S, model, test):
        verboseprint("Computing permutation for FDR")
        np.random.seed(seed=self.options.seed)
        #tests = int(1/self.options.fdr)
        tests = 10

        self.ppermute = np.zeros((tests, S))

        for ps in range (tests):
            self.snps_permute = self.snps[np.random.choice(N, N, 
                replace=False), :]
            if self.options.mode == "multitrait":
                pvalues_permute, betas_permute, NLL0, NLLAlt, model = \
                        self.run_mt_anyeffect_GWAS(N, P, S, test, 
                                computeFDR=True)
                self.ppermute[ps, :] = pvalues_permute
            if self.options.mode == "singletrait":
                pvalues_permute, pvalues_permute_adjust, betas_permute, 
                stats_permute, z_permute, model = self.run_st_GWAS(N, P, S, 
                        test, computeFDR=True)
                pvalues_adjust_min = pvalues_permute_adjust.min(axis=0)
                self.ppermute[ps, :] = pvalues_adjust_min
        
        SNPsPassingFDR = int(self.options.fdr*S*tests)
        allppermute = self.ppermute.flatten()
        allppermute.sort()
        self.FDR = allppermute[SNPsPassingFDR]
        
        outstring = (self.options.output, model, self.options.chromosome, 
                self.options.fileend)
	pd.DataFrame(self.ppermute).to_csv("%s/%s_ppermute_%s%s.csv" % 
                outstring, index=False)
	pd.DataFrame(['FDR', str(self.FDR)]).T.to_csv(
                "%s/%s_empiricalFDR_%s%s.csv" % outstring, 
                header=False, index=False)
        
        return(self)


    ##############
    ### output ###
    ##############
	
    def manhattanQQ(self, model, P, colorS='DarkBLue', colorNS='Orange', 
            alphaNS=0.05, thr_plotting=0.05):
	self.position, chromBounds = self.getCumSum(self.position)
	fig = plt.figure(figsize=[12,4])
	ax1 = fig.add_subplot(2,1,1)
        
        if self.options.computeFDR:
            thr_plotting = self.FDR
            self.options.fileend = "%s_%s%s" % (self.options.fileend, 
                    "FDR", self.options.fdr)
        if self.options.mode == 'singletrait':
            pv = self.pvalues_adjust.T.min(axis=1).ravel()
        if self.options.mode == 'multitrait':
            pv = self.pvalues.ravel() 
        
        plot.plot_manhattan(posCum=self.position['pos_cum'].values.astype(int), 
                pv=pv, colorS=colorS, colorNS=colorNS, alphaNS=alphaNS, 
                thr_plotting=thr_plotting)
	ax1.set_title('%s' % self.options.chromosome)
	ax2 = fig.add_subplot(2,1,2)
	plot.qqplot(self.pvalues.ravel())
        fig.tight_layout()
	fig.savefig('%s/%s_%s%s.png' % (self.options.output, 
            self.options.chromosome, model, self.options.fileend))
	return self

    def getCumSum (self, offset=100000, chrom_len=None):
	RV = self.position.copy()
        # sp.unique is always sorted
	chromvals = sp.unique(self.position['chrom'])
        #get the starting position of each Chrom
	chrom_pos_cum = sp.zeros_like(chromvals)
	pos_cum = sp.zeros_like(self.position.shape[0])
	offset = 100000
	if not 'pos_cum' in self.position:
            #get the cum_pos of each variant.
	    RV["pos_cum"] = sp.zeros_like(self.position['pos'])
	pos_cum = RV['pos_cum'].values
	maxpos_cum = 0
	for i,mychrom in enumerate(chromvals):
	    chrom_pos_cum[i] = maxpos_cum
	    i_chr=self.position['chrom'] == mychrom
	    if chrom_len is None:
		maxpos = self.position['pos'][i_chr].values.astype(int).max() \
                        + offset
	    else:
		maxpos = chrom_len[i] + offset
	    pos_cum[i_chr.values] = maxpos_cum + \
                    self.position.loc[i_chr,'pos'].values.astype(int)
	    maxpos_cum += maxpos
	return (RV, chrom_pos_cum)

    def writeResult(self, model, CHR, SNP, POS, columns=None, thr=5e-8):
        outstring = (self.options.output, model, self.options.chromosome, 
                self.options.fileend)

	if self.options.mode == 'singletrait':
	    beta_df = pd.DataFrame(self.betas.T, index=SNP, columns=columns)
	    stats_df = pd.DataFrame(self.stats.T, index=SNP, columns=columns)
	    pvalue_df = pd.DataFrame(self.pvalues.T, index=SNP, 
                    columns=columns)
	    pvalues_adjust_df = pd.DataFrame(self.pvalues_adjust.T, index=SNP, 
                    columns=columns)
	    z_df = pd.DataFrame(self.z.T, index=SNP, columns=columns)
	    pmin_df = pd.DataFrame(self.pvalues.T.min(axis=1), index=SNP, 
                    columns=['Pmin'])
	    padjust_min_df = pd.DataFrame(self.pvalues_adjust.T.min(axis=1), 
                    index=SNP, columns=['Pmin'])
	    

	    pvalue_df['CHR'] = CHR
	    pvalue_df['POS'] = POS
	    pvalue_df['SNP'] = SNP
	    
            pvalues_adjust_df['CHR'] = CHR
	    pvalues_adjust_df['POS'] = POS
	    pvalues_adjust_df['SNP'] = SNP

	    pmin_df['CHR'] = CHR
	    pmin_df['POS'] = POS
	    pmin_df['SNP'] = SNP
            
            padjust_min_df['CHR'] = CHR
	    padjust_min_df['POS'] = POS
	    padjust_min_df['SNP'] = SNP
	    
	    beta_df['CHR'] = CHR
	    beta_df['POS'] = POS
	    beta_df['SNP'] = SNP

	    stats_df['CHR'] = CHR
	    stats_df['POS'] = POS
	    stats_df['SNP'] = SNP

	    z_df['CHR'] = CHR
	    z_df['POS'] = POS
	    z_df['SNP'] = SNP
	    
	    cols = pvalue_df.columns.tolist()
	    cols = cols[len(cols)-3:len(cols)] + cols[:-3]

	    beta_df = beta_df[cols]
	    pvalue_df = pvalue_df[cols]
	    pvalues_adjust_df = pvalues_adjust_df[cols]
	    stats_df = stats_df[cols]
	    z_df = z_df[cols]

            if self.pvalues_empirical is not None:
                cols_emp = cols[0:3]
	        cols_emp.extend(['Pempirical'])
	        
                pempirical_df = pd.DataFrame(self.pvalues_empirical.T, 
                        index=SNP, columns=['Pempirical'])
	        pempirical_df['CHR'] = CHR
	        pempirical_df['POS'] = POS
	        pempirical_df['SNP'] = SNP
	        pempirical_df = pempirical_df[cols_emp]
	        
                pempirical_raw_df = pd.DataFrame(self.pvalues_empirical_raw.T, 
                        index=SNP, columns=columns)
	        pempirical_raw_df['CHR'] = CHR
	        pempirical_raw_df['POS'] = POS
	        pempirical_raw_df['SNP'] = SNP
	        pempirical_raw_df = pempirical_raw_df[cols]
	    
            cols=cols[0:3]
	    cols.extend(['Pmin'])
	    pmin_df = pmin_df[cols]
	    padjust_min_df = padjust_min_df[cols]
	    psig_df = pmin_df.loc[pmin_df['Pmin'] < 5e-8]

	    if pvalue_df.shape[1] !=4:
		pmin_df.to_csv("%s/%s_pminvalue_%s%s.csv" % outstring, 
                        index=False)
		padjust_min_df.to_csv("%s/%s_padjust_minvalue_%s%s.csv" % 
                        outstring, index=False)
	    pvalues_adjust_df.to_csv("%s/%s_padjust_%s%s.csv" % outstring, 
                    index=False)
	    psig_df.to_csv("%s/%s_psigvalue_%s%s.csv" % outstring, index=False)
	    stats_df.to_csv("%s/%s_statsvalue_%s%s.csv" % outstring, 
                    index=False)
	    z_df.to_csv("%s/%s_zvalue_%s%s.csv" % outstring, index=False)
	    
        else:
	    beta_df = pd.DataFrame(self.betas.T, index=SNP, columns=columns)
	    pvalue_df = pd.DataFrame(SNP, index=SNP, columns=['SNP'])

	    beta_df['SNP'] = SNP
	    beta_df['CHR'] = CHR
	    beta_df['POS'] = POS
	    
            cols = beta_df.columns.tolist()
	    cols = cols[len(cols)-3:len(cols)] + cols[:-3]

	    beta_df = beta_df[cols]
	    
            pvalue_df['CHR'] = CHR
	    pvalue_df['POS'] = POS
	    pvalue_df['P'] = self.pvalues.flatten()

            if self.pvalues_empirical is not None:
	        pempirical_df = pd.DataFrame(SNP, index=SNP, columns=['SNP'])
	        pempirical_df['CHR'] = CHR
	        pempirical_df['POS'] = POS
	        pempirical_df['P'] = self.pvalues_empirical

            if self.options.likelihoods:
	        NLL0_df = pd.DataFrame(SNP, index=SNP, columns=['SNP'])
	        NLL0_df['CHR'] = CHR
	        NLL0_df['POS'] = POS
	        NLL0_df['NLL'] = self.NLL0.T
	        NLLAlt_df = pd.DataFrame(SNP, index=SNP, columns=['SNP'])
	        NLLAlt_df['CHR'] = CHR
	        NLLAlt_df['POS'] = POS
	        NLLAlt_df['NLL'] = self.NLLAlt.T
	
        pvalue_df.to_csv("%s/%s_pvalue_%s%s.csv" % outstring, index=False)
	beta_df.to_csv("%s/%s_betavalue_%s%s.csv" % outstring, index=False)
        if self.pvalues_empirical is not None:
            pempirical_df.to_csv("%s/%s_pempirical_%s%s%s.csv" % 
                    (outstring + (self.options.fdr,)), index=False)
            
            if self.pvalues_empirical_raw is not None:
                pempirical_raw_df.to_csv("%s/%s_pempirical_raw%s%s%s.csv" % 
                        (outstring + (self.options.fdr,)), index=False)
        if self.options.likelihoods:
            NLL0_df.to_csv("%s/%s_NLL0_%s%s.csv" % outstring, index=False)
            NLLAlt_df.to_csv("%s/%s_NLLAlt_%s%s.csv" % outstring, index=False)
	
        return 0


#########################
### data manipulation ###
#########################

def boolanize(string):
    """ Convert command line parameter "True"/"False" into bool"""
    return bool(strtobool(string))

def scale(x):
    x = x - np.array(x.mean(axis=0), dtype=float)
    x /= x.std(axis=0)
    return x

def verboseprint(message, verbose=True):
    if verbose is True:
	print message

def match(samples_ref, samples_compare, data_compare, squarematrix=False):
    samples_before=samples_compare.shape[0]
    subset = pd.match(samples_ref, samples_compare)

    data_compare = data_compare[subset,:]
    if squarematrix:
    	data_compare = data_compare[:, subset]
    samples_compare = samples_compare[subset]
    samples_after=samples_compare.shape[0]
    np.testing.assert_array_equal(samples_ref, samples_compare,  
            err_msg="Col order does not match. These are the differing columns:\t%s" % ( np.array_str( np.setdiff1d( samples_ref, 
                samples_compare ))))
    return (data_compare, samples_compare, samples_before, samples_after)

def AlleleFrequencies(snp):
        hc_snps = np.array([makeHardCalledGenotypes(s) for s in snp])
        counts = np.array(np.unique(hc_snps, return_counts=True))
        frequencies = counts[1,:]/float(len(hc_snps))
        major_a = sqrt(frequencies.max())
        minor_a = 1 - major_a
        return minor_a, major_a

def makeHardCalledGenotypes(snp):
    if snp <= 0.5:
        return 0
    elif snp > 1.5:
        return 2
    else:
        return 1

def adjustForMeff(pv, Meff):
    pvadjust = np.array([min(pveff, 1) for pveff in (pv * Meff)])
    return pvadjust

############
### main ###
############

def main():
    # create data object
    dataparse= DataParse()
    #datainput.getArgs(debug=True)
    dataparse.getArgs()


    # getting data for GWAS analyses
    #pdb.set_trace()
    datainput = DataInput(options=dataparse.options)
    datainput.getPhenotypes()
    datainput.transform()
    datainput.getGenotypes()
    datainput.getKinship()
    datainput.getCovariates()
    datainput.getVD()
    datainput.getPCs()

    # transforming/subsetting/matching data
    datainput.commonSamples()
    if datainput.options.freqonly is True:
        datainput.getAlleleFrequencies()
    else:
       # pdb.set_trace()
        datainput.subsetTraits()
        datainput.regress_and_transform()

        ## create GWAS object
        datagwas = DataGWAS(datainput, options=dataparse.options)
        datagwas.run_GWAS()
   
    ### running analyses ###

if __name__ == "__main__":
       main()





