from limmbo.io.parser import ParseData
from limmbo.io.reader import ReadData
from limmbo.io.input import InputData
from limmbo.core.vdbootstrap import LiMMBo

def entry_point():

    # parse command-line arguments
    dataparse = ParseData()
    dataparse.getArgs()

    # read data specified in command-line arguments
    dataread = ReadData(verbose=dataparse.options.verbose)
    dataread.getPhenotypes(file_pheno = dataparse.options.file_pheno)
    dataread.getCovariates(file_covariates = dataparse.options.file_covariates)
    dataread.getRelatedness(file_relatedness = dataparse.options.file_relatedness)
    
    # combine all input, check for consistency and pre-process data
    datainput = InputData(verbose=dataparse.options.verbose)
    datainput.addPhenotypes(phenotypes = dataread.phenotypes,
                            phenotype_ID = dataread.phenotype_ID,
                            pheno_samples = dataread.pheno_samples)
    datainput.addRelatedness(relatedness = dataread.relatedness,
                            relatedness_samples = dataread.relatedness_samples)
    datainput.addCovariates(covariates = dataread.covariates,
                            covs_samples = dataread.covs_samples)
    datainput.commonSamples()
    datainput.subsetTraits(traitstring = dataparse.options.traitstring)
    datainput.regress(regress = dataparse.options.regress)
    datainput.transform(type = dataparse.options.transform)

    # set up variance decomposition via LiMMBo
    datalimmbo = LiMMBo(datainput=datainput,
            S=dataparse.options.S, 
            timing=dataparse.options.timing,
            iterations=dataparse.options.iterations,
            verbose=dataparse.options.verbose)
    resultsBS = datalimmbo.runBootstrapCovarianceEstimation(
            seed=dataparse.options.seed, cpus=dataparse.options.cpus, 
            minCooccurrence=dataparse.options.minCooccurrence, 
            n=dataparse.options.runs)

    resultsCovariance = datalimmbo.combineBootstrap(results=resultsBS) 
    datalimmbo.saveVarianceComponents(resultsCovariance,
            output=dataparse.options.output,
            intermediate=dataparse.options.intermediate)


#############
### main  ###
#############

if __name__ == "__main__":
    entry_point()
