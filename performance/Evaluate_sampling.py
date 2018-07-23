import scipy as sp
import numpy as np
import pandas as pd
import _pickle as pickle

import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns


def multiple_set_covers_all(number_of_traits, sample_size_ori, number_of_covers,
        seed=2152):
    #Compute a set of subsets that represent a multiple set covers
    number_to_tuple = dict()
    global_counter = 0

    #Create a dictionary to store mappings between tuples and traits. Should be a function really
    for i in range(0, number_of_traits):
        for j in range(i+1, number_of_traits):
            number_to_tuple[global_counter] = (i,j)
            global_counter = global_counter +1

    number_of_trait_tuples = (number_of_traits*(number_of_traits-1))//2

    inflated_sample_size = sample_size_ori-1

    goal_set_cover_size = number_of_trait_tuples//inflated_sample_size +1
    used_subsets = list()

    set_cover = range(1, inflated_sample_size+1)
    #Compute the set covers one at a time
    for i in range(1 ,number_of_covers+1):
        for j in range(1, goal_set_cover_size+1):
            new_set_cover = [((x+(i-1)+(j-1)*inflated_sample_size) % number_of_trait_tuples) for x in set_cover]
            used_subsets.append(np.array(new_set_cover))

    flattened_list = list()
    #Map back from tuple index to trait ids
    count=1
    for set_tuple in used_subsets:
        list_to_flat = [number_to_tuple[x] for x in set_tuple]
        flattened_list.append(np.array([item for sublist in list_to_flat for item in sublist]))
        count = count + 1
    #Remove duplicates from each subset
    bootstrap_array=list([np.unique(xi) for xi in flattened_list])
    length_array = [len(x) for x in bootstrap_array]

    maxlen = max(length_array)

    #Because of the mapping from traits to tuples the method might not always pick each set to be
    #the same size. It seemed like that was needed by the method so the below code randomly fixes 
    #the unequal set size. If the number of tuples is close to the sample size this will be slow

    rand_state = np.random.RandomState(seed)
    counts = sp.zeros((number_of_traits, number_of_traits))
    for i in range(0,len(bootstrap_array)):
        while (len(bootstrap_array[i])!=maxlen):
            bootstrap_array[i] = np.unique(np.append(rand_state.choice(
                    a=list(range(number_of_traits)),
                    size=(maxlen-len(bootstrap_array[i])),
                    replace=False),
                bootstrap_array[i]))
        index = np.ix_(np.array(bootstrap_array[i]), np.array(bootstrap_array[i]))
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
        sampling = multiple_set_covers_all(number_of_traits=P, sample_size_ori=S,
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
    m=3
    P=[50, 100, 500, 1000]
    S=10
    seed = range(1,51)
    directory = '/homes/hannah/python_modules/limmbo/performance'
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
