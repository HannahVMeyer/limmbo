import scipy as sp
import numpy as np
import pandas as pd


def multiple_set_covers_all(number_of_traits, sample_size_ori, number_of_covers,
        seed=2152):
    #Compute a set of subsets that represent a multiple set covers
    number_to_tuple = dict()
    global_counter = 0

    #Create a dictionary to store mappings between tuples and traits. Should be a function really
    for i in range(1, number_of_traits):
        for j in range(i+1, number_of_traits):
            number_to_tuple[global_counter] = (i,j)
            global_counter = global_counter +1

    number_of_trait_tuples = (number_of_traits*(number_of_traits-1))//2

    inflated_sample_size = sample_size_ori-1

    goal_set_cover_size = number_of_trait_tuples//inflated_sample_size +1
    used_subsets = list()

    set_cover = range(1, inflated_sample_size+1)
    #Compute the set covers one at a time
    for i in range(1,number_of_covers+1):
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
    for i in range(0,len(bootstrap_array)):
        while (len(bootstrap_array[i])!=maxlen):
            bootstrap_array[i] = np.unique(np.append(rand_state.choice(
                    a=list(range(number_of_traits)),
                    size=(maxlen-len(bootstrap_array[i])),
                    replace=False),
                bootstrap_array[i]))
    return bootstrap_array

def random_sampling(P, S, minCooccurrence, seed=2152):
    rand_state = np.random.RandomState(seed)
    counts = sp.zeros((P, P))
    return_list = []
    while counts.min() < minCooccurrence:
        bootstrap = rand_state.choice(a=list(range(P)), size=S, replace=False)
        return_list.append(bootstrap)
        index = np.ix_(np.array(bootstrap), np.array(bootstrap))
        counts[index] += 1
    return(np.array(return_list))

m=3
P=100
S=10
seed=243
test_rs = random_sampling(P=P, S=S, minCooccurrence=m, seed=243)
test_sc = multiple_set_covers_all(number_of_traits=P, sample_size_ori=S,
        number_of_covers=m, seed=seed)
