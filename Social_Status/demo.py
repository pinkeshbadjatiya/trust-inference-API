import numpy as np
import math
from collections import OrderedDict
import os
import sys
import itertools

# trust_hTrust = np.load("G_final_hTrust")
# trust_sTrust = np.load("G_final_sTrust")
# trust_hsTrust = np.load("G_final_hsTrust")

G = np.load("self_G.npy")
G_original = np.load("self_G_original.npy")


# G = np.random.random((10,10))
# G_original = np.asarray([[1,2],[5,6],[5,7],[2,4]])

model = input("Enter model here: ")
print model

def TP_accuracy(model,G,G_original):
    
    if model == "hTrust":
        G_final = np.load("G_final_hTrust.npy")
        print "MADE IT!"
    if model == "sTrust":
        G_final = np.load("G_final_sTrust.npy")
    if model == "hsTrust":
        G_final = np.load("G_final_hsTrust.npy")

    print "TOTAL NO USERS: 2216"
    print "Percentage of total data: 10%"


    # set ratio of data to split
    print "Splitting data........"

    q = 0.5
    # r,c = np.where(G > 0)
    # n_N = int(len(zip(r,c)) *0.5)
    # N = zip(r,c)[n_N:]
    # print "N: ", len(N)

    A = G_original

    n_N = int(len(G_original) *q) #getting middle number
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
    
    print "X = ", q, "Prediction Accuracy= ", PA1
    return PA1

TP_accuracy(model,G,G_original)




