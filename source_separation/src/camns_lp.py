import numpy as np
import random


class camns_object:
    def __init__(self):
        self.vector__ = None
        self.size__ = None
    

def get_random_observations(sources, observ_num=None):
    """
    Sources: (L, N)
    A: (N, M)               # random matrix
    X = S @ A: (L, M)
    """
    sources_num = sources.shape[1]
    if observ_num is None:
        observ_num = sources_num

    random_matrix = np.random.random((sources_num, observ_num))
    column_sum = np.sum(random_matrix, axis=1).reshape(-1, 1)
    random_matrix /= column_sum
    return sources @ random_matrix