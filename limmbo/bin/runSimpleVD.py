######################
### import modules ###
######################

import h5py

from limmbo.io.parser import DataParse
from limmbo.io.input import DataInput
from limmbo.core.vdsimple import simpleCD

#################
### functions ###
#################
def entry_point():

    # initiate DataParse object
    dataparse = DataParse()
    dataparse.getArgs()

    # datareader = DataReader()

    # initiate Data objects
    datainput = DataInput()


    ### running analyses ###
    datainput.getPhenotypes()
    datainput.getRelatedness()
    #datainput.commonSamples()
    datainput.subsetTraits()
    datainput.getCovariates()
    datainput.regress_and_transform()
    
    Cg, Cn, processtime = simpleVD(datainput = datainput, 
            cache = dataparse.options.cache,
            output = dataparse.options.output,
            iterations = dataparse.options.iterations,
            verbose = dataparse.options.verbose)

    # save predicted covariance matrics
    try:
        pd.DataFrame(Cg).to_csv('{}/Cg_REML.csv'.format(
            dataparse.options.output), sep=",", header=False, index=False)
        pd.DataFrame(Cn).to_csv('{}/Cn_RML.csv'.format(
            dataparse.options.output), sep=",", header=False, index=False)
        pd.DataFrame([processtime]).to_csv('{}/process_time_REML.',
            '.csv'.format(dataparse.options.output), sep=",", header=False,
            index=False)
    except:
        raise IOError('Cannot write to {}: check writing permissions',
            '{}'.format(dataparse.options.output))


############
### main ###
############

if __name__ == "__main__":
    entry_point()
