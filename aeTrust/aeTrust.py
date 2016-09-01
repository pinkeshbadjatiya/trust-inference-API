#! /usr/bin/env python

''' Training module (mini-batch SGD) '''

import numpy as np
import theano
from theano import tensor as T
from model import AutoEncoder
import pdb
import resource
from scipy.special import expit
import os
import gc


CURRENT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "./")
PATH = os.path.abspath(CURRENT_DIR + "data/data.mat")

class trainAE(object):

    def __init__(self, path, k, lr= 0.01, batch_size=1, loss='bce', n_epochs=500):
        '''
        Arguments:
            path : path to training data
            k : hidde unit's dimension
            lr : learning rate
            batch_size : batch_size for training, currently set to 1
            loss : loss function (bce, rmse) to train AutoEncoder
            n_epochs : number of epochs for training
        '''
        self.AE = AutoEncoder(path, k)
        # Definne the autoencoder model
        #self.AE.model()
        self.AE.model_batch()
        self.epochs = n_epochs


    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))


    # Batch training method
    def train(self, batch_size):
        T = self.AE.T
        T = T.tocsr()
        nonzero_indices = T.nonzero()
        #pdb.set_trace()
        n_users = len(np.unique(nonzero_indices[0]))
        indices = np.unique(nonzero_indices[0])
        for epoch in xrange(self.epochs):
            l = []
            for ind, i in enumerate(xrange(0, n_users, batch_size)):
                ratings = T[indices[i:(i + batch_size)], :].toarray().astype(np.float32)
                loss = self.AE.ae_batch(ratings)
                l.append(loss)
            m = np.mean(np.array(l))
            print("mean Loss for epoch %d  batch %d is %f"%(epoch, ind, m))
            rmse = self.RMSE()
            print("RMSE after one epoch is %f"%(rmse))

    def RMSE(self):
        W, V, b, mu = self.AE.get_params()
        print("testing process starts")
        test = self.AE.test_ind
        rat = []
        pred = []
        rmse = 0
        for i,j in test:
            Rt = self.AE.T[i, :].todense()
            Rt1 = np.zeros(Rt.shape[1])
            Rt1[:] = Rt[:]
            #pdb.set_trace()
            p = expit(np.dot(W, expit(np.dot(V, Rt1) + mu)) + b)
            #p = np.tanh(np.dot(W, np.tanh(np.dot(V, Rt1) + mu)) + b)
            p = p[j]
            pred.append(p)
            rat.append(self.AE.t[i, j])
        try:
            rat = np.array(rat)
            pred = np.array(pred)
            rmse = np.sqrt(np.mean((pred-rat)**2))
        except:
            print "exception"
            pdb.set_trace()

        return rmse
        #pdb.set_trace()


def main():
    autoencoder = trainAE(CURRENT_DIR + 'data/data.mat', 500)
    autoencoder.train(32)
    #autoencoder.RMSE_sparse()
    autoencoder.RMSE()




if __name__ == "__main__":
    main()

