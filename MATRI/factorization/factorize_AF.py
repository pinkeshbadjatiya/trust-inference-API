""" Alternating Factorization """
import numpy as np
from globs import *
import copy
from numpy.linalg import norm


def alternatinUpdate(P, F, G, neighbour_list, num_factors):
    """ Factorize the given matrix using the alternatingUpdate algorithm
        mentioned in MATRI-report
    """
    F1 = copy.deepcopy(F)
    for i in neighbour_list.keys():
        # set of column indices
        a = neighbour_list[i]
        d = np.zeros((len(a), 1))
        G1 = np.zeros((len(a), num_factors))
        for j in xrange(len(a)):
            d[j] = P[i, a[j]]
            G1[j, :] = G[a[j], :]
        # Vectorize previous loop
        # d = self.T[i, a]
        # 1[xrange(len(a)), :] = G[a,:]
        # TO-DO: Use sklearn's regression to find F1[i, :] instead
        G1t = G1.T
        temp = np.linalg.inv((np.dot(G1t, G1) + GLOBAL_lamda * np.eye(num_factors)))
        F1[i, :] = np.dot(np.dot(temp, G1t), d).reshape(num_factors,)
    return F1



def mat_fact_AF(X, T_uni_dim, num_factors, k):
    """ Factorization code from the paper
        Returns:    dim(F) = nxr  || dim(G.T) = nxr
    """
    #F0 = np.random.uniform(low=0.05, high=0.15, size=(self.T.shape[0], self.r))
    #G0 = np.random.uniform(low=0.05, high=0.15, size=(self.T.shape[0], self.r))
    F0 = np.random.rand(T_uni_dim, num_factors)
    G0 = np.random.rand(T_uni_dim, num_factors)
    #F0 = np.zeros((self.T.shape[0], self.r))
    #G0 = np.zeros((self.T.shape[0], self.r))
    #F0[:] = 1/float(self.r)
    #G0[:] = 1/float(self.r)
    # pre-process self.k for alternatingUpate
    d = {}
    for i,j in k:
        if not d.has_key(i):
            d[i] = []
        d[i].append(j)
    iter = 1
    E1 = -1
    E2 = -1

    while iter < GLOBAL_FACTORIZATION_MAX_ITER:
        F = alternatinUpdate(X, F0, G0, d, num_factors)
        G = alternatinUpdate(X.T, G0, F, d, num_factors)

        if iter == 1:
            iter += 1
            F0[:] = F[:]
            G0[:] = G[:]
            continue

        E1 = norm(F-F0)
        E2 = norm(G-G0)
        if E1 <= GLOBAL_EPS and E2 <= GLOBAL_EPS:
            break
        F0[:] = F[:]
        G0[:] = G[:]
        iter += 1

    # d = List of (List of all the neighbours of a node whose trust values are known)
    return F0, G0, d
