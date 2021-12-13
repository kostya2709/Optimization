import numpy as np
import scipy as sp
from scipy import stats
from numpy import random
import cvxpy as cp
from cvxopt import matrix, solvers

solvers.options['show_progress'] = False

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

    random_matrix = np.random.rand(observ_num, sources_num)
    column_sum = np.sum(random_matrix, axis=1, keepdims=True)
    random_matrix /= column_sum
    return sources @ random_matrix.T

def is_extreme_point(C, d, alpha, tol):
    vec = C @ alpha + d
    L, D = C.shape
    rows = np.all(np.abs(vec) < tol, axis=1)
    T = C[rows, :]
    if T.shape[0] == 0:
        return False
    return np.linalg.matrix_rank(T, tol=tol) == D

def remove_zeros(observs):
    # TODO: Implement
    return observs

def camns_lp(observs, sources_num=None):
    """
    X is the L-by-M observation matrix, where M is the number of
    observations.
    N is the number of sources. 
    """
    TOL_LP= 1e-3;       # tolerance for (small) numerical errors in LP
    TOL_EXT= 1e-6;      # tolerance for extreme-point validation
    TOL_ZEROS= 1e-6;    # tolerance for eliminating zero observation points
    
    if sources_num is None:
        sources_num = observs.shape[1]

    X = remove_zeros(observs)
    L, M = X.shape
    N = sources_num
    d = np.mean(X, axis=1, keepdims=True)
    [C, Sigma, V] = np.linalg.svd(X - d, full_matrices=False)
    C= C[:, :N-1]
    el = 0
    Q1 = np.zeros((L, 1))
    S = np.zeros((L, N))
    lp_cnt = 0
    E_L = np.eye(L)
    while el < N:
        B = E_L - Q1 @ Q1.T
        w = stats.norm.rvs(size=L)
        r = B @ w
        c = matrix(-C.T @ r)
        A_ub = matrix(-C)
        b_ub = matrix(d)

        solution = solvers.conelp(c, A_ub, b_ub)
        alpha1 = np.array(solution['x'])
        opt_vec1 = C @ alpha1 + d
        p_star = r.T @ (opt_vec1)
        lp_cnt += 1

        solution = solvers.conelp(-c, A_ub, b_ub)
        alpha2 = np.array(solution['x'])
        opt_vec2 = C @ alpha2 + d
        q_star = r.T @ (opt_vec2)
        lp_cnt += 1

        if el == 0:
            if is_extreme_point(C, d, alpha1, TOL_EXT):
                S[:, el : el+1] = opt_vec1
                el += 1
            if is_extreme_point(C, d, alpha2, TOL_EXT):
                S[:, el : el+1] = opt_vec2
                el += 1
        else:
            p_star_norm = np.linalg.norm(p_star)
            q_star_norm = np.linalg.norm(q_star)
            r_norm = np.linalg.norm(r)
            if p_star_norm / (r_norm * np.linalg.norm(opt_vec1)) >= TOL_EXT:
                if is_extreme_point(C, d, alpha1, TOL_EXT):
                    S[:, el : el+1] = opt_vec1
                    el += 1
            if q_star_norm / (r_norm * np.linalg.norm(opt_vec2)) >= TOL_EXT:
                if is_extreme_point(C, d, alpha2, TOL_EXT):
                    S[:, el : el+1] = opt_vec2
                    el += 1

        if el > 0:
            Q1, R = np.linalg.qr(S)

        print("lp_cnt:", lp_cnt)
    return S
    