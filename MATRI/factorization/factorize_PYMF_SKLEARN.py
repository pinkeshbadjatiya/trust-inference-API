from __future__ import division
import networkx as nx
from scipy.io import savemat
import json
import numpy as np
import pdb
from sklearn.decomposition import NMF
import numpy
import pymf

def mat_fact_SKLEARN(X, r):
    # ISSUE : sklearn's NMF accepts only non-neg matrices.
    model = NMF(n_components=r, init='nndsvda', alpha=0.001)
    L = model.fit_transform(X)
    H = model.components_
    return L, H.T


def mat_fact_PYMF(X, r):
    """ X - matrix, l - latent factors
        Returns 2 factors of X, such that, dim(L) = n x r
                                           dim(R.T) = r x n
    """
    # Using pymf factorization
    nmf = pymf.NMF(X, num_bases=r)#, niter=200)
    nmf.factorize()
    L = nmf.W
    R = nmf.H
    return L, R.T
