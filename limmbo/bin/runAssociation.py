from limmbo.io.parser import getGWASargs
from limmbo.io.reader import ReadData
from limmbo.io.input import InputData
from limmbo.core.gwas import GWAS
from limmbo.core.utils import _biallelic_dosage
from tqdm import tqdm

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
        file_genotypes=options.file_genotypes, delim=options.genotypes_delim,
        file_samples=options.file_samples)
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
    datainput.addPhenotypes(phenotypes=dataread.phenotypes)
    datainput.addGenotypes(genotypes=dataread.genotypes,
                           genotypes_info=dataread.genotypes_info,
                           geno_samples=dataread.genotypes_samples,
                           genotypes_darray=dataread.genotypes_darray)
    if traitlist is not None:
        datainput.subsetTraits(traitlist=traitlist)
    if dataread.relatedness is not None:
        datainput.addRelatedness(relatedness=dataread.relatedness)
    if dataread.covariates is not None:
        datainput.addCovariates(covariates=dataread.covariates)
    if dataread.Cg is not None:
        datainput.addVarianceComponents(Cg=dataread.Cg, Cn=dataread.Cn)
    datainput.commonSamples(samplelist=samplelist)
    if options.regress:
        datainput.regress()
    if options.transform is not None:
        datainput.transform(transform=options.transform)

    # set up variance decomposition via LiMMBo
    gwas = GWAS(datainput=datainput,
                verbose=options.verbose)

    mode = 'singletrait' if options.singletrait else 'multitrait'
    setup = 'lmm' if options.lmm else 'lm'

    if datainput.genotypes_darray:
        chunks = datainput.genotypes.chunks[0]
        start = 0
        header = True
        writemode = 'w'
        for c in tqdm(chunks, desc="Association", disable=not gwas.verbose):
            end = start + c

            geno_chunk = datainput.genotypes[start:end,:].compute()
            gwas.genotypes = _biallelic_dosage(geno_chunk).T
            gwas.genotypes_info = gwas.genotypes_info.iloc[start:end,:]

            resultsAssociation = gwas.runAssociationAnalysis(
                setup=setup,
                mode=mode,
                adjustSingleTrait=options.adjustP)
            gwas.saveAssociationResults(resultsAssociation,
                    outdir=options.outdir, name=options.name, mode=writemode,
                    header=header)

            if options.plot:
                if start == 0:
                    pvalues = resultsAssociation['pvalues']
                    if mode == 'singletrait':
                        pvalues_adjust = resultsAssociation['pvalues_adjust']
                else:
                    pvalues = np.concatenat(pvalues,
                            resultsAssociation['pvalues'])
                    if mode == 'singletrait' and options.adjustP:
                        pvalues_adjust = np.concatenat(pvalues_adjust,
                                resultsAssociation['pvalues_adjust'])

            header=False
            writemode = 'a'
            start = end

        if options.plot:
            if mode == 'singletrait' and options.adjustP:
                resultsAssociation = {'pvalues': pvalues,
                        'pvalues_adjust': pvalues_adjust}
            else:
                resultsAssociation = {'pvalues': pvalues}
    else:
        resultsAssociation = gwas.runAssociationAnalysis(
            setup=setup,
            mode=mode,
            adjustSingleTrait=options.adjustP)

        if options.nrpermutations is not None:
            pvalues_empirical = gwas.computeEmpiricalP(
                pvalues=resultsAssociation['pvalues'],
                seed=options.seed,
                nrpermutations=options.nrpermutations)
        else:
            pvalues_empirical = None

        if options.fdr is not None:
            empirical_fdr, empirical_pvalue_dist = gwas.computeFDR(
                fdr=options.fdr,
                seed=options.seed)

        gwas.saveAssociationResults(resultsAssociation, outdir=options.outdir,
            name=options.name,
            pvalues_empirical=pvalues_empirical)

    if options.plot:
        if gwas.fdr_empirical is not None:
            thr = gwas.fdr_empirical
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
