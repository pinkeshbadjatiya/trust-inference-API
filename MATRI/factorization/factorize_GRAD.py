from __future__ import print_function
import theano
import sys
from theano import tensor as T
import numpy as np
from sklearn.decomposition import NMF
import pdb

#def ref_mat_fact(X, r):
#    P = T.matrix()
#    ind = T.neq(P, 0).nonzero()
#    #Ind = theano.tensor.matrix(ind[0], ind[1])
#    def step(index_i,index_j, P):
#        return P[index_i, index_j]
#
#    res, _ = theano.scan(fn=step, sequences=[ind[0], ind[1]], non_sequences=[P])
#
#    f = theano.function([P], res)
#
#    X = np.array([[1,0,0,1],[-1,0,0,1],[1,1,0,0],[-1,0,0,1]]).astype(np.float32)
#    print(f(X))


def mat_fact_GRAD(X, r):
    X = X.astype(np.float32)
    f = np.random.rand(X.shape[0], r)*0.1
    g = np.random.rand(X.shape[0], r)*0.1
    #f = np.zeros((X.shape[0], r))
    #g = np.zeros((X.shape[0], r))
    #f[:] = 1/float(r)
    #g[:] = 1/float(r)
    F = theano.shared(f)
    G = theano.shared(g)
    params = [F, G]
    P = T.matrix()
    lamda = 0.1
    alpha = 0.5
    ind = T.neq(P, 0).nonzero()

    def step(index_i, index_j, P, F, G):
        temp = (P[index_i, index_j] - T.dot(F[index_i, :], G[index_j, :].T)) ** 2
        return temp

    res, _ = theano.scan(fn=step, sequences=[ind[0], ind[1]], non_sequences=[P, F, G])
    loss = T.mean(res) + lamda * (T.sum(F ** 2 ) + T.sum(G ** 2))
    grads = T.grad(loss, params)
    updates = [(param, param - alpha * grad) for param, grad in zip(params, grads)]
    factorize = theano.function([P], loss, updates=updates)
    for i in xrange(100):
        print("\r LOSS:", factorize(X), end="")
        sys.stdout.flush()
    print("")
    return F.get_value(), G.get_value()




if __name__ == "__main__":
    X = np.random.rand(30,30).astype(np.float32)
    F1, G1  = mat_fact_GRAD(X,2)
    model = NMF(n_components=2,init='random')
    F = model.fit_transform(X)
    G = model.components_
    print(X)
    print(np.dot(F1, G1.T))
    print(np.dot(F, G))
    pdb.set_trace()
