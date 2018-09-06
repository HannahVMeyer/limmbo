import scipy as sp
import numpy as np
import pandas as pd
import _pickle as pickle
import os
import random

import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

#The set cover of tuples is considered as trying to cover a square matrix of size
def find_square(i_coord, j_coord, side_length):
    return np.union1d(np.array(range(i_coord, i_coord+side_length)), np.array(range(j_coord, j_coord+side_length)))

#Relabel a subset so we don't need to recompute set covers for large m
def relabel_subset(subset, permutation):
    return [permutation[x] for x in subset]

########
#This method computes multiple set covers of tuples. 
#To do so we consider instead covering a matrix is size number_of_traits*number_of_traits.   
########
def multiple_set_covers_all(number_of_traits, sample_size, number_of_covers,
        seed=2152):
    sample_size_t = sample_size
    if (sample_size % 2) != 0:#We make odd sized samples even and then fix the sampling size later.
        sample_size_t = sample_size -1
    
    #Here we compute one set cover. After that we can generate the others based on this by relabelling
    #the matrix rows/cols with a permutation
    i = 0 
    j = 0
    used_subsets = list()
    while (j < number_of_traits):
        while (i < number_of_traits):
            if i==j:#We're on the main diagonal and we can get a better set than normal. We can cover .
                used_subsets.append(find_square(i,j,sample_size_t) % number_of_traits)
                i += sample_size_t
            else:#We're not on the main diagonal so we can only cover a square of size sample_size/2*sample_size/2  
                used_subsets.append(find_square(i,j,sample_size_t//2) % number_of_traits)
                i += (sample_size_t//2)
        j += sample_size_t//2
        if j%sample_size_t ==0:
            i=j
        else:
            i = (j//sample_size_t)*sample_size_t +sample_size_t#Set i to new start position            
    
    counts = sp.zeros((number_of_traits, number_of_traits))
    #Don't bother recomputing the set cover just relabel in randomly. 
    #(you can do it in a non-random way but the coverage will look cluster around the main diagonal
    if (sample_size % 2) !=0 :
        for i in range(0,len(used_subsets)):
            used_subsets[i] = np.append(used_subsets[i], np.random.randint(0,number_of_traits))
    
    bootstrap_array = list()
    for num in range(0,number_of_covers):
        order = np.random.permutation(number_of_traits)    
        for i in range(0,len(used_subsets)):
            bootstrap_array.append(relabel_subset(used_subsets[i], order))
            index = np.ix_(relabel_subset(used_subsets[i], order),
                relabel_subset(used_subsets[i], order))
            counts[index] += 1
    return {'bootstrap': bootstrap_array, 'counts': counts}

def random_sampling(P, S, minCooccurrence, seed=2152):
    rand_state = np.random.RandomState(seed)
    counts = sp.zeros((P, P))
    return_list = []
    while counts.min() < minCooccurrence:
        bootstrap = rand_state.choice(a=list(range(P)), size=S, replace=False)
        return_list.append(bootstrap)
        index = np.ix_(np.array(bootstrap), np.array(bootstrap))
        counts[index] += 1
    return {'bootstrap': np.array(return_list), 'counts': counts}

def drawTraits(P, S, minCooccurrence, method, seed):
    if method == 'set':
        sampling = multiple_set_covers_all(number_of_traits=P, sample_size=S,
            number_of_covers=minCooccurrence, seed=seed)
    if method == 'random':
        sampling = random_sampling(P=P, S=S, minCooccurrence=minCooccurrence,
                seed=seed)
    nrSamples = len(sampling['bootstrap'])
    return  nrSamples, sampling['counts']

def evaluate(trait_list, S, minCooccurrence, method, seed_list, verbose=True):
    counts = {}
    ptime = {}
    samples = {}
    for trait in trait_list:
        counts[trait] = nans((len(seed_list),trait,trait))
        samples[trait] =  nans(len(seed_list))
        ptime[trait] =  nans(len(seed_list))
        seed_count = 0
        for seed in seed_list:
            if verbose:
                print("Seed {} for trait {}".format(seed, trait))
            t0 = time.clock()
            nrSamples, c = drawTraits(trait, S, minCooccurrence, method, seed)
            t1 = time.clock()
            ptime[trait][seed_count] = t1 - t0
            samples[trait][seed_count] = nrSamples
            counts[trait][seed_count,:,:] = c
            seed_count = seed_count + 1
    return counts, ptime, samples

def nans(shape):
    a = np.empty(shape, dtype=float)
    a.fill(np.nan)
    return a

if __name__ == "__main__":
    m=4
    P=[50, 100, 300, 500]
    S=15
    seed = range(1,21)
    directory = os.environ['HOME'] +'/python_modules/limmbo/performance'
    c_set, t_set, s_set = evaluate(P, S, m, 'set', seed)
    c_random, t_random, s_random  = evaluate(P, S, m, 'random', seed)
    pickle.dump({'c_set': c_set, "t_set": t_set, "s_set": s_set,
        'c_random': c_random, "t_random": t_random, "s_random": s_random},
        open("{}/CompareRandom2SetCoverage.p".format(directory), "wb" ))


    with PdfPages('{}/sampling.pdf'.format(directory)) as pdf:
        # count distribution
        gs = mpl.gridspec.GridSpec(nrows=2, ncols=5, wspace=0.05,
                width_ratios=(3,3,3,3,1))
        fig = plt.figure(figsize=(15, 6))
        for trait in range(len(P)):

            trait_set = c_set[P[trait]].mean(axis=0)
            trait_set /= trait_set.max()
            _ax = fig.add_subplot(gs[0, trait])
            plt.imshow(trait_set)
            _ax.axis('off')
            _ax.set_title('{} traits'.format(P[trait]), fontsize=10)

            if not trait:
                _ax.text(-0.1, 0.5, transform=_ax.transAxes, s='Set cover',
                        ha='right', va='center', rotation=90)

            trait_random = c_random[P[trait]].mean(axis=0)
            trait_random /= trait_random.max()
            _ax = fig.add_subplot(gs[1, trait])
            plt.imshow(trait_random)
            _ax.axis('off')

            if not trait:
                _ax.text(-0.1, 0.5, transform=_ax.transAxes, s='Random sampling',
                        ha='right', va='center', rotation=90)

        cax = fig.add_subplot(gs[:, -1])
        plt.colorbar(ax=cax, shrink=0.5, fraction=1)
        cax.axis('off')
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # sampling and time
        gs = mpl.gridspec.GridSpec(nrows=len(P), ncols=2, wspace=0.3)
        fig = plt.figure(figsize=(7,14))

        for trait in range(len(P)):
            df_random = pd.DataFrame(data=np.array(s_random[P[trait]]).T,
                    columns=['NrSamples'])
            df_random['Method'] = 'Random sampling'
            df_set = pd.DataFrame(data=np.array(s_set[P[trait]]).T,
                    columns=['NrSamples'])
            df_set['Method'] = 'Set cover'
            df = df_random.append(df_set)

            _ax = fig.add_subplot(gs[trait,0])
            sns.boxplot(x="Method", y="NrSamples", data=df, ax=_ax)
            _ax.set_ylabel('')
            if not trait:
                _ax.set_title('Number of Samples', fontsize=10)
            if trait == len(P)-1:
                _ax.set_xticklabels(labels=['Random sampling', 'Set cover'])
            else:
                _ax.set_xlabel('')
                _ax.set_xticklabels("")
                _ax.set_xticks([])
            _ax.text(-0.3, 0.5, transform=_ax.transAxes, s='{} traits'.format(
                P[trait]), ha='right', va='center')

            df_random = pd.DataFrame(data=np.array(t_random[P[trait]]).T,
                    columns=['Time'])
            df_random['Method'] = 'Random sampling'
            df_set = pd.DataFrame(data=np.array(t_set[P[trait]]).T,
                    columns=['Time'])
            df_set['Method'] = 'Set cover'
            df = df_random.append(df_set)

            _ax = fig.add_subplot(gs[trait,1])
            sns.boxplot(x="Method", y="Time", data=df, ax=_ax)
            _ax.set_ylabel('')
            if not trait:
                _ax.set_title('Process time [s]', fontsize=10)

            if trait == len(P)-1:
                _ax.set_xticklabels(labels=['Random sampling', 'Set cover'])
            else:
                _ax.set_xlabel('')
                _ax.set_xticklabels("")
                _ax.set_xticks([])

        pdf.savefig(bbox_inches='tight')
        plt.close()
