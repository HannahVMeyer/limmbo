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
    dataread.getGenotypes(file_genotypes = dataparse.options.file_genotypes)
    
    # combine all input, check for consistency and pre-process data
    datainput = InputData(verbose=dataparse.options.verbose)
    datainput.addPhenotypes(phenotypes = dataread.phenotypes,
                            phenotype_ID = dataread.phenotype_ID,
                            pheno_samples = dataread.pheno_samples)
    datainput.addGenotypes(genotypes = dataread.genotypes,
                            genotypes_info = dataread.genotypes_info,
                            geno_samples = dataread.geno_samples)
    datainput.addRelatedness(relatedness = dataread.relatedness,
                            relatedness_samples = dataread.relatedness_samples)
    datainput.addCovariates(covariates = dataread.covariates,
                            covs_samples = dataread.covs_samples)
    datainput.commonSamples()
    datainput.subsetTraits(traitstring = dataparse.options.traitstring)
    datainput.regress(regress = dataparse.options.regress)
    datainput.transform(type = dataparse.options.transform)

    # set up variance decomposition via LiMMBo
    gwas = GWAS(datainput=datainput,
            seed=dataparse.options.seed,
            searchDelta=dataparse.options.searchDelta,
            verbose=dataparse.options.verbose)
            
    resultsAssociation = gwas.runAssociationAnalysis(
        setup=dataparse.options.setup,
        mode=dataparse.options.mode,
        adjustSingleTrait = dataparse.options.adjustP)

    if dataparse.options.nrpermutations is not None:
        resultsEmpiricalP = gwas.computeEmpiricalP(nrpermutations =
            dataparse.options.nrpermutations)
    
    if dataparse.options.fdr is not None:
       resultsFDR = gwas.__computeFDR(fdr = dataparse.options.fdr)

    if plot:
        if gwas.fdr_empirical is not None:
            thr = gwas.fdr.empirical
        else:
            thr = dataparse.options.thr

        output = '{}/{}_{}.png' % (output, chromosome, model)
        manhattanQQ(pvalues=resultsAssociation['pvalues'], 
            colorS=dataparse.options.colorS, 
            colorNS=dataparse.options.colorNS,
            alphaNS=dataparse.options.alphaNS, 
            thr_plotting=thr, savePlot = output)


#############
### main  ###
#############

if __name__ == "__main__":
    entry_point()
