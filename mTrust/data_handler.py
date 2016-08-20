#! /usr/bin/env python

import numpy as np
from scipy.io import loadmat
import pdb

class data_handler():
    
    def __init__(self, rating_path, trust_path):
        self.rating_path = rating_path
        self.trust_path = trust_path
        self.n_users = 0
        self.n_prod = 0
        self.n_cat = 6
    
    def get_stats(self):
        return self.n_users, self.n_prod, self.n_cat

    def load_matrices(self):
        # Loading Matrices from data
        f1 = open(self.rating_path)
        f2 = open(self.trust_path)
        R = loadmat(f1)
        W = loadmat(f2)
        # Converting R and W from dictionary to array
        R = R['rating_with_timestamp']
        #R = R['rating']
        W = W['trust']
        self.n_users = max(R[:, 0])
        self.n_prod = max(R[:, 1])
        # Selecting entries with the 6 categories given in the paper
        cat_id = [7, 8, 9, 10, 11, 19]
        cat_map = {7:0, 8:1, 9:2, 10:3, 11:4, 19:5}
        #cat_id = [8, 14, 17, 19, 23, 24]
        #cat_map = {8:0, 14:1, 17:2, 19:3, 23:4, 24:5}
        R = R[np.in1d(R[:, 2],cat_id)]
        R = R[R[:, 5].argsort()]
        R_size = R.shape[0]
        # Choosing 70% data for training and rest for testing
        R_train = R[:R_size*0.7]
        R_test = R[R_size*0.7:]
        # Making all eligible Product-Category pairs
        ones = np.ones(R_train.shape[0])
        prod_cat = dict(zip(zip(R_train[:, 1], R_train[:, 2]), ones))
        # Making the mu matrix
        mu = np.zeros(6)
        for cat in cat_id:
            cat_rating = R_train[np.where(R_train[:, 2] == cat), 3]
            mu[cat_map[cat]] = np.mean(cat_rating)
        return R_train, R_test, W, prod_cat, mu
            
if __name__ == "__main__":
    data = data_handler("../data/rating_with_timestamp.mat", "../data/trust.mat")
    R_train, R_test, W, PF_pair, mu = data.load_matrices()
    print "done"
