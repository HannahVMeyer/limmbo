from limmbo.io.parser import getGWASargs
from limmbo.io.reader import ReadData
from limmbo.io.input import InputData
from limmbo.core.vdbootstrap import LiMMBo

def entry_point():

    # parse command-line arguments
    parser = getGWASargs()
    options = parser.parse_args()

    # read data specified in command-line arguments
    dataread = ReadData(verbose=options.verbose)
    dataread.getPhenotypes(file_pheno = options.file_pheno)
    dataread.getCovariates(file_covariates = options.file_covariates)
    dataread.getRelatedness(file_relatedness = options.file_relatedness)
    dataread.getGenotypes(file_genotypes = options.file_genotypes)
    
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
    datainput.subsetTraits(traitstring = options.traitstring)
    datainput.regress(regress = options.regress)
    datainput.transform(type = options.transform)

    # set up variance decomposition via LiMMBo
    gwas = GWAS(datainput=datainput,
            seed=options.seed,
            searchDelta=options.searchDelta,
            verbose=options.verbose)
           
    if options.singletrait:
        mode='singletrait'
    else:
        mode='multitrait'

    if options.lmm: 
        setup='lmm'
    else:
        setup='lm'

    resultsAssociation = gwas.runAssociationAnalysis(
        setup=setup,
        mode=mode,
        adjustSingleTrait = options.adjustP)

    if options.nrpermutations is not None:
        pvalues_empirical = gwas.computeEmpiricalP(nrpermutations =
            options.nrpermutations)
    else:
        pvalues_empirical = None
    
    if options.fdr is not None:
       empirical_fdr, empirical_pvalue_dist = gwas.computeFDR(fdr = options.fdr)

    saveAssociationResults(resultsAssociation, outdir=options.outdir,
            name=options.name, pvalues_empirical = pvalues_empirical)
    if plot:
        if gwas.fdr_empirical is not None:
            thr = gwas.fdr.empirical
        else:
            thr = options.thr

        output = '{}/{}_{}.png' % (output, chromosome, model)
        manhattanQQ(pvalues=resultsAssociation, 
            colorS=options.colorS, 
            colorNS=options.colorNS,
            alphaNS=options.alphaNS, 
            thr_plotting=thr, savePlot = options.outdir)


#############
### main  ###
#############

if __name__ == "__main__":
    entry_point()
