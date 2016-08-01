from trial_data import data_handler
import numpy as np
import math
from collections import OrderedDict
import os
import sys



CURRENT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "./")
_rating_path = os.path.abspath(CURRENT_DIR + "data/rating_with_timestamp.mat")
_trust_path = os.path.abspath(CURRENT_DIR + "data/trust.mat")
_trust_with_timestamp_path = os.path.abspath(CURRENT_DIR + "data/epinion_trust_with_timestamp.mat")



def calcS(P):
    #calculating matrix S or homophily coefficient matrix using formula in paper

    # Z = np.load(fname + '.npy')
    n = len(P)
    numerator = np.dot(P, P.transpose())

    print "made numerator"

    J_norm = (np.sum(np.abs(P)**2,axis=-1)**(1./2)).astype(np.float32)
    # J_norm = J_norm.astype(np.float32)
    J_norm = J_norm.reshape(n,1)
    I_norm = J_norm.transpose().astype(np.float32)
    denominator = np.dot(J_norm,I_norm)

    print "made denominator"

    # dividing numerator and denominator and handling setting all nan to zero

    with np.errstate(divide='ignore', invalid='ignore'):
        Z = np.divide(numerator,denominator)
        Z = np.nan_to_num(Z)

    print "made Z"

    return Z


def hTrust(G, S, _lambda, K, maxIter, P, G_original):
    beta = 0.01
    alpha = 0.01


    # construct L using formula in paper
    # Summing up elements of each row
    d = [np.sum(x) for x in S]
    D = np.diag(d).astype(np.float32) # D = Diagonal matrix
    L = np.subtract(D,S).astype(np.float32) # L = Laplacian matrix


    [n, n] = G.shape


    # Initialize U and V randomly

    U = np.random.uniform(0.0,0.1,(n,K))
    V = np.random.uniform(0.0,0.1,(K, K))

    # U = np.ones((n,K)) * 0.05
    # V = np.ones((K,K)) * 0.05

    # U = np.load("U.npy")
    # V = np.load("sigma.npy")

    # U = np.random.random((n, K))
    # V = np.random.random((K, K))

    iter = 0

    #% Hamid: Main loop
    while (iter < maxIter):

        UU = np.dot(U.T, U)

        A = np.dot(U, np.dot(V.T, np.dot(UU, V))) + np.dot(U,np.dot(V,np.dot(UU, V.T))) + alpha * U + _lambda * np.dot(D, U) + 1e-8
        B = np.dot(np.dot(G.T, U),V) + np.dot(np.dot(G, U),V.T) + _lambda * np.dot(S, U)
        U = U * np.sqrt(B / A)

        AV = np.dot(np.dot(UU, V), UU) + beta * V + 1e-8
        BV = np.dot(np.dot(U.T, G), U)
        V = V * np.sqrt(BV/ AV)

        # Obj = np.linalg.norm((G - np.dot(np.dot(U, V),U.T)), ord =('fro')) ** 2 + alpha * np.linalg.norm(U, ord = ('fro')) ** 2 + beta * np.linalg.norm(V, ord = ('fro')) ** 2 + _lambda * np.trace(np.dot(np.dot(U.T, L) * U))
        print (('the object is in iter %d is %f'), iter)

        iter = iter + 1


    GC = np.dot(np.dot(U, V),U.T)

    np.save("G_final_hTrust",GC)

    print "TP accuracy: " + str(TP_accuracy(G_original, G, GC))


    print "FINAL MATRIX CALCULATED"
    # print GC
    return GC

def TP_accuracy(G_original, G, GC):

    G_final = GC


    q = 0.5
    # r,c = np.where(G > 0)
    # n_N = int(len(zip(r,c)) *0.5)
    # N = zip(r,c)[n_N:]
    # print "N: ", len(N)

    A = G_original

    n_N = int(len(G_original) *0.5) #getting middle number
    print n_N
    N = G_original[n_N:] #getting N as np array
    N = [tuple(x) for x in N] #turning N to a list of tuples

    print "N: ", len(N)
    # print N


    # # N = random.sample(zip(r,c), n_N)
    # r,c = np.where(G == 0)
    # # B = random.sample(zip(r,c), n_N)
    # B = zip(r,c)
    # print "B: ", len(B)

    A = set([tuple(x) for x in A]) #turning A into a set
    all_pairs = [(a,b) for ((a,b),z) in np.ndenumerate(G)]
    all_pairs = set(all_pairs)
    B = list(all_pairs-A)
    print "B PAIRS", len(B)

    print type(B), type(N)

    BUN = B + N
    T1 = []

    print "Union: ", len(BUN)

    for u, v in BUN:
        ptrust = G_final[u-1,v-1]
        T1.append((ptrust, u, v))

    T1.sort()
    T1.reverse()

    cnt1 = 0

    for i in xrange(n_N):
        if (T1[i][1],T1[i][2]) in N:
            cnt1 += 1

    print "CNT: ", cnt1
    print n_N
    PA1 = float(cnt1) / len(N)
    print PA1

    del BUN
    del N
    del B
    del T1

    print "X = ", q, "Prediction Accuracy (Betweenness nx)= ", PA1
    return PA1


def init_main():
    data = data_handler(_rating_path, _trust_path, _trust_with_timestamp_path)
    data = data.load_matrices()
    K = data[2]

    P = data[0]
    print "set p" + str(len(P))
    G = data[1]
    print "set G" + str(len(G))
    G_original = data[3]
    _lambda = 10
    print "LOADED MATRICES"

    S = calcS(P)
    print hTrust(G,S,_lambda, K, 50, P, G_original)



if __name__ == "__main__":
    init_main()
