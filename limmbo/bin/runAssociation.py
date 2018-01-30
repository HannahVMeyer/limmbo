from limmbo.io.parser import getGWASargs
from limmbo.io.reader import ReadData
from limmbo.io.input import InputData
from limmbo.core.gwas import GWAS


def entry_point():

    # parse command-line arguments
    parser = getGWASargs()
    options = parser.parse_args()
    if options.lmm and options.file_relatedness is None:
        parser.error("For --lmm, --kf is required")
    if options.file_covariates is None and \
            options.file_pcs is None and options.regress is True:
        parser.error(("Regress is set to True but neither covariate file",
                      "nor PC file provided"))

    # read data specified in command-line arguments
    dataread = ReadData(verbose=options.verbose)
    dataread.getPhenotypes(file_pheno=options.file_pheno,
                           delim=options.pheno_delim)
    dataread.getGenotypes(
        file_genotypes=options.file_genotypes, delim=options.genotypes_delim)
    if options.file_covariates is not None:
        dataread.getCovariates(file_covariates=options.file_covariates,
                               delim=options.covariate_delim)
    if options.file_relatedness is not None:
        dataread.getRelatedness(file_relatedness=options.file_relatedness,
                                delim=options.relatedness_delim)
    if options.file_cg is not None:
        dataread.getVarianceComponents(file_Cg=options.file_cg,
                                       delim_cg=options.cg_delim,
                                       file_Cn=options.file_cn,
                                       delim_cn=options.cn_delim)
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
    datainput.addPhenotypes(phenotypes=dataread.phenotypes,
                            phenotype_ID=dataread.phenotype_ID,
                            pheno_samples=dataread.pheno_samples)
    datainput.addGenotypes(genotypes=dataread.genotypes,
                           genotypes_info=dataread.genotypes_info)
    if traitlist is not None:
        datainput.subsetTraits(traitlist=traitlist)
    if dataread.relatedness is not None:
        datainput.addRelatedness(relatedness=dataread.relatedness)
    if dataread.covariates is not None:
        datainput.addCovariates(covariates=dataread.covariates)
    if dataread.Cg is not None:
        datainput.addVarianceComponents(Cg=dataread.Cg, Cn=dataread.Cn)
    datainput.commonSamples(samplelist=samplelist)
    if options.regress is not None:
        datainput.regress()
    if options.transform is not None:
        datainput.transform(transform=options.transform)

    # set up variance decomposition via LiMMBo
    gwas = GWAS(datainput=datainput,
                seed=options.seed,
                verbose=options.verbose)

    if options.singletrait:
        mode = 'singletrait'
    else:
        mode = 'multitrait'

    if options.lmm:
        setup = 'lmm'
    else:
        setup = 'lm'

    resultsAssociation = gwas.runAssociationAnalysis(
        setup=setup,
        mode=mode,
        adjustSingleTrait=options.adjustP)

    if options.nrpermutations is not None:
        pvalues_empirical = gwas.computeEmpiricalP(
            nrpermutations=options.nrpermutations)
    else:
        pvalues_empirical = None

    if options.fdr is not None:
        empirical_fdr, empirical_pvalue_dist = gwas.computeFDR(fdr=options.fdr)

    gwas.saveAssociationResults(resultsAssociation, outdir=options.outdir,
                                name=options.name,
                                pvalues_empirical=pvalues_empirical)

    if options.plot:
        if gwas.fdr_empirical is not None:
            thr = gwas.fdr.empirical
        else:
            thr = options.thr

        gwas.manhattanQQ(results=resultsAssociation,
                         colourS=options.colourS,
                         colourNS=options.colourNS,
                         alphaS=options.alphaS,
                         alphaNS=options.alphaNS,
                         thr_plotting=thr,
                         saveTo=options.outdir)


if __name__ == "__main__":
    entry_point()
