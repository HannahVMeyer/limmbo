from limmbo.io.parser import getVarianceEstimationArgs
from limmbo.io.reader import ReadData
from limmbo.io.input import InputData
from limmbo.core.vdbootstrap import LiMMBo

def entry_point():

    # parse command-line arguments
    parser = getVarianceEstimationArgs()
    options = parser.parse_args()
    if options.file_covariates is None and options.file_pcs is None and \
        options.regress is True:
            parser.error(("Regress is set to True but neither covariate file",
            "nor PC file provided"))
    # read data specified in command-line arguments
    dataread = ReadData(verbose=options.verbose)
    dataread.getPhenotypes(file_pheno = options.file_pheno)
    dataread.getCovariates(file_covariates = options.file_covariates)
    dataread.getRelatedness(file_relatedness = options.file_relatedness)
    
    # combine all input, check for consistency and pre-process data
    datainput = InputData(verbose=options.verbose)
    datainput.addPhenotypes(phenotypes = dataread.phenotypes,
                            phenotype_ID = dataread.phenotype_ID,
                            pheno_samples = dataread.pheno_samples)
    datainput.addRelatedness(relatedness = dataread.relatedness,
                            relatedness_samples = dataread.relatedness_samples)
    datainput.addCovariates(covariates = dataread.covariates,
                            covs_samples = dataread.covs_samples)
    datainput.commonSamples()
    datainput.subsetTraits(traitstring = options.traitstring)
    datainput.regress(regress = options.regress)
    datainput.transform(type = options.transform)

    # set up variance decomposition via LiMMBo
    datalimmbo = LiMMBo(datainput=datainput,
            S=options.S, 
            timing=options.timing,
            iterations=options.iterations,
            verbose=options.verbose)
    resultsBS = datalimmbo.runBootstrapCovarianceEstimation(
            seed=options.seed, cpus=options.cpus, 
            minCooccurrence=options.minCooccurrence, 
            n=options.runs)

    resultsCovariance = datalimmbo.combineBootstrap(results=resultsBS) 
    datalimmbo.saveVarianceComponents(resultsCovariance,
            output=options.output,
            intermediate=options.intermediate)


#############
### main  ###
#############

if __name__ == "__main__":
    entry_point()
