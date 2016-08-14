#! /usr/bin/env python

from __future__ import print_function
import numpy as np
from theano import tensor as T
import theano
from scipy.sparse import coo_matrix
import pdb
from scipy.io import loadmat
from lasagne.updates import sgd, adagrad, adam, apply_momentum


theano.config.compute_test_value = 'off'

class AutoEncoder(object):

    def __init__(self, path, k):
        # hidden layer's dimension
        self.k = k
        self.t = loadmat(path)
        self.t = self.t['data']
        nz_ind = self.t.nonzero()
        NZ = np.vstack((nz_ind[0], nz_ind[1])).T
        np.random.shuffle(NZ)
        self.train_ind = NZ[:len(NZ)-500]
        self.test_ind = NZ[len(NZ)-500:]
        t = self.t.tolil()
        self.gt = self.t[self.test_ind[:, 0], self.test_ind[:, 1]]
        t[self.test_ind[:, 0], self.test_ind[:,1]] = 0
        self.T = t.tocsc()
        #pdb.set_trace()
        self.n = self.T.shape[0]
        self.W = []
        self.V = []
        self.b = []
        self.mu = []

    def model_batch(self, loss='bce', lr=0.001):
        ''' define the AutoEncoder model (mini-batch) '''
        # initializing network paramters
        w = np.random.uniform(low=-np.sqrt(6 / float(self.n + self.k)),
                              high=np.sqrt(6 / float(self.n + self.k)),
                              size=(self.n, self.k)).astype(np.float32)
        v = np.random.uniform(low=-np.sqrt(6 / float(self.n + self.k)),
                              high=np.sqrt(6 / float(self.n + self.k)),
                              size=(self.k, self.n)).astype(np.float32)
        MU = np.zeros((self.k)).astype(np.float32)
        B = np.zeros((self.n)).astype(np.float32)
        # Creating theano shared variables from these
        W = theano.shared(w, name='W', borrow=True)
        V = theano.shared(v, name='V', borrow=True)
        mu = theano.shared(MU, name='mu', borrow=True)
        b = theano.shared(B, name='b', borrow=True)
        self.W = W
        self.V = V
        self.b = b
        self.mu = mu
        self.param = [W, V, mu, b]
        # input variable (matrix) with each row corresponding to each user's
        # vector of indices of observed values
        # TO-DO : Check if this batch thing works and if not try out with vector
        #index = T.imatrix()
        # Ratings correspoding to the indices
        rating = T.matrix()
        #rating.tag.test_value = np.array([[1,0,0],[0,0,1],[1,1,1]]).astype(np.int32)
        # Target for the network (ratings only)
        # TO-DO : Check for aliasing issue, if any
        target = rating
        ######################## Theano scan part ##############################
        def step(rat, W, V, mu, b):
            # Function applied at each step of scan
            # find all non-zero indices from index (observed values)
            #ind_nz = T.gt(ind, 0).nonzero()
            #rat.tag.test_value = np.array([1,0,0]).astype(np.int32)
            res = T.zeros_like(rat)
            rat_nz = T.neq(rat, 0).nonzero()[0]
            #print(rat_nz.tag.test_value.shape)
            # target
            # Check for mem. aliasing
            tar = rat[rat_nz]
            #hidden_activation = T.tanh(T.dot(V[:, rat_nz], rat[rat_nz]) #\
            #                           + mu)
            hidden_activation = T.nnet.sigmoid(T.dot(V[:, rat_nz], rat[rat_nz]) #\
                                       + mu)

            #print(hidden_activation.tag.test_value.shape)
            #hidden_activation = hidden_activation.reshape((hidden_activation.shape[0], hidden_activation[1]))
            output_activation = T.nnet.sigmoid(T.dot(W[rat_nz, :], \
                                             hidden_activation) \
                                       + b[rat_nz])
            #print(output_activation.tag.test_value.shape)

            #print(output_activation.shape.eval(rat=np.array([1,0,0]).astype(np.int32)))
            res = T.set_subtensor(res[rat_nz] ,output_activation)
            #T.set_subtensor(res[rat_nz], output_activation, inplace=True)
            #return T.sum(output_activation)
            #return hidden_activation
            return res
            #return rat_nz

        scan_res, scan_updates = theano.scan(fn=step, outputs_info=None, \
                                             sequences=[rating],
                                             non_sequences=[W, V, \
                                                            mu, b])

        # NO need for mean. T.sum will sum across the whole matrix
        #self.loss = T.mean(T.sum((scan_res - rating) ** 2), axis=1)
        self.loss = T.sum((scan_res - rating) ** 2) + \
            0.001 * T.sum(W ** 2) + 0.001 * T.sum(V ** 2)
        #updates_sgd = sgd(self.loss, self.param, learning_rate=lr)
        #updates = apply_momentum(updates_sgd, self.param, momentum=0.9)
        #updates_adagrad = adagrad(self.loss, self.param, learning_rate=lr)
        updates_adam = adam(self.loss, self.param, learning_rate=lr)
        #grads = T.grad(self.loss, self.param)
        #updates = [(param, param - lr * grad) for (param, grad) in \
        #           zip(self.param, grads)]

        self.ae_batch = theano.function([rating], self.loss, updates=updates_adam)
        self.debug = theano.function([rating], scan_res, updates=None)

    def model(self, lr = 0.4, loss='rmse'):
        ''' Define model on single example (batch_size = 1) '''
        W = np.random.uniform(low=-np.sqrt(6 / float(self.n + self.k)),
                              high=np.sqrt(6 / float(self.n + self.k)),
                              size=(self.n, self.k))
        V = np.random.uniform(low=-np.sqrt(6 / float(self.n + self.k)),
                              high=np.sqrt(6 / float(self.n + self.k)),
                              size=(self.k, self.n))
        #mu = np.zeros((self.k, 1))
        #b = np.zeros((self.n, 1))
        mu = np.zeros((self.k,))
        b = np.zeros((self.n,))
        # Creating theano shared variables from these
        self.W = theano.shared(W, name='W', borrow=True)
        self.V = theano.shared(V, name='V', borrow=True)
        self.mu = theano.shared(mu, name='mu', borrow=True)
        self.b = theano.shared(b, name='b', borrow=True)
        # Stack all parameters in a single array
        self.param = [self.W, self.V, self.mu, self.b]
        # input variable, vector corresponding to user i's trust
        # Vector of observed indices in the training data
        index = T.ivector()
        # Ratings correspoding to the observed indices
        # NOTE: Try with both (0,1) and (-1,1) thing
        rating = T.ivector()
        # Target for the network (target = input)
        # TO-DO : Check for aliasing issue, if any
        target = rating
        # Calculate hidden layer activations h =  (g(Vr + mu))
        hidden_activation = T.tanh(T.dot(self.V[:, index], rating)
                                   + self.mu)
        # Calculate output layer activations ( f(Wh + b))
        output_activation = T.tanh(T.dot(self.W[index, :], hidden_activation)
                                   + self.b[index])
        # Compute the loss
        if loss == 'bce':
            # binary cross-entropy loss
            self.loss = - T.sum(target * T.log(output_activation) + (1 - target)
                                * T.log(1 - output_activation))
        elif loss == 'rmse':
            # RMSE loss
            self.loss = T.sum((target - output_activation) ** 2) + \
                        0.001 * T.sum(self.W ** 2) + 0.001 * T.sum(self.V ** 2)

        # gradients w.r.t paramters
        grads = T.grad(self.loss, wrt=self.param)
        updates = [(param, param - lr * param_grad) for (param, param_grad)
                   in zip(self.param, grads)]
        self.ae = theano.function(inputs=[index, rating],
                                     outputs=self.loss, updates=updates)


    def get_params(self):
        return self.W.get_value(), self.V.get_value(), self.b.get_value(), self.mu.get_value()

if __name__ == "__main__":
    AE = AutoEncoder('../data/data.mat', 100)
    AE.model_batch()
    rating = np.array([[1,1,1],[1,1,1],[1,1,1]]).astype(np.float32)
    AE.ae_batch(rating)
    #x = AE.debug(rating)
    pdb.set_trace()
