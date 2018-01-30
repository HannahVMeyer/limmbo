from limmbo.io.parser import getVarianceEstimationArgs
from limmbo.io.reader import ReadData
from limmbo.io.input import InputData
from limmbo.core.vdbootstrap import LiMMBo
from limmbo.core.vdsimple import vd_reml

import pandas as pd


def entry_point():

    # parse command-line arguments
    parser = getVarianceEstimationArgs()
    options = parser.parse_args()
    if options.file_covariates is None and options.regress is True:
        parser.error(("Regress is set to True but no covariate file",
                      "provided"))
    # read data specified in command-line arguments
    dataread = ReadData(verbose=options.verbose)
    dataread.getPhenotypes(file_pheno=options.file_pheno,
                           delim=options.pheno_delim)
    dataread.getRelatedness(
        file_relatedness=options.file_relatedness,
        delim=options.relatedness_delim)
    if options.file_covariates is not None:
        dataread.getCovariates(file_covariates=options.file_covariates,
                               delim=options.covariate_delim)
    if options.samplelist is not None or options.file_samplelist is not None:
        samplelist = dataread.getSampleSubset(
                samplelist=options.samplelist,
                file_samplelist=options.file_samplelist)
    else:
        samplelist = None
    if options.traitstring is not None:
        traitlist = dataread.getTraitSubset(traitstring=options.traitstring)
    else:
        traitlist = None

    # combine all input, check for consistency and pre-process data
    datainput = InputData(verbose=options.verbose)
    datainput.addPhenotypes(phenotypes=dataread.phenotypes)
    datainput.addRelatedness(relatedness=dataread.relatedness)
    if dataread.covariates is not None:
        datainput.addCovariates(covariates=dataread.covariates)
        datainput.regress()
    datainput.commonSamples(samplelist=samplelist)
    datainput.subsetTraits(traitlist=traitlist)
    if options.transform is not None:
        datainput.transform(transform=options.transform)

    if options.limmbo:
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
                                          output=options.outdir,
                                          intermediate=options.intermediate)

    if options.reml:
        Cg, Cn, processtime = vd_reml(datainput=datainput,
                                      iterations=options.iterations,
                                      verbose=options.verbose)
        try:
            pd.DataFrame(Cg).to_csv('{}/Cg_REML.csv'.format(
                options.outdir), sep=",", header=False, index=False)
            pd.DataFrame(Cn).to_csv('{}/Cn_REML.csv'.format(
                options.outdir), sep=",", header=False, index=False)
            pd.DataFrame([processtime]).to_csv(
                '{}/process_time_REML.csv'.format(
                    options.outdir), sep=",", header=False, index=False)
        except:
            raise IOError(('Cannot write to {}: check writing '
                           'permissions').format(options.outdir))


if __name__ == "__main__":
    entry_point()
