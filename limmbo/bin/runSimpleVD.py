from limmbo.io.parser import ParseData
from limmbo.io.reader import ReadData
from limmbo.io.input import InputData
from limmbo.core.vdsimple import vd_reml

import pandas as pd

def entry_point():

    # initiate DataParse object
    dataparse = ParseData()
    dataparse.getArgs()

    # datareader = DataReader()
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
    
    Cg, Cn, processtime = vd_reml(datainput = datainput, 
            iterations = dataparse.options.iterations,
            verbose = dataparse.options.verbose)

    # save predicted covariance matrics
    try:
        pd.DataFrame(Cg).to_csv('{}/Cg_REML.csv'.format(
            dataparse.options.output), sep=",", header=False, index=False)
        pd.DataFrame(Cn).to_csv('{}/Cn_REML.csv'.format(
            dataparse.options.output), sep=",", header=False, index=False)
        pd.DataFrame([processtime]).to_csv('{}/process_time_REML.csv'.format(
	    dataparse.options.output), sep=",", header=False, index=False)
    except:
        raise IOError('Cannot write to {}: check writing permissions'.format(
			dataparse.options.output))


############
### main ###
############

if __name__ == "__main__":
    entry_point()
