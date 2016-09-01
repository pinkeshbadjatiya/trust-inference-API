import numpy as np
import math
import operator
from operator import itemgetter
from collections import OrderedDict
import sys
import itertools
import gc
import random

#this program loads the U,H,G_original,and G matrices created after running sTrust_2.py and calculates accuracy.


U = np.load("self_U.npy")
print "SIZE OF U ", sys.getsizeof(U)

print "BUT TOTAL USERS", len(U)

H = np.load("self_H.npy")
print "SIZE OF H ", sys.getsizeof(H)

G_original = np.load("self_G_original.npy")
print "SIZE OF G_original", sys.getsizeof(G_original)

G = np.load("self_G.npy")
print "SIZE OF G ", sys.getsizeof(G)

# U = np.random.random((220,220))
# print "SIZE OF U ", sys.getsizeof(U)

# H = np.random.random((220,220))
# print "SIZE OF H ", sys.getsizeof(H)

# G = np.random.random((220,220))
# print "SIZE OF G ", sys.getsizeof(G)

# G_original = np.asarray([[5,3],[1,1],[7,8],[1,2]])
# print "SIZE OF G_original", sys.getsizeof(G_original)

# print "LOADED ALL MATRICES"


def calcTrust(U,H,G,G_original):
        #calculate all final trust values knowing U,H

        G_final = np.dot(U,H)
        G_final = np.dot(G_final, U.transpose())
        print "SIZE OF U ", sys.getsizeof(U)

        print "CALCULATED FINAL TRUST MATRIX"

        np.save("G_final_sTrust",G_final)

        print "Found predicted trust! Has TP accuracy: " + str(calculateAccuracy(G_final,G,G_original))


        print "FINAL PREDICTED"
        print G_final
        
        # print "START OFF AS"
        # print G

        return G_final



def TP_accuracy(G_final,G,G_original):
    # set ratio of data to split
        

        ratio = int(len(G_original) *0.5)
        
        A = np.asarray(G_original)

        print "CREATED A of size"
        print sys.getsizeof(A)
    
        # C = A[:ratio]
        # print C
        D = np.delete(A,[x for x in range(0,ratio)],0)
        
        print "CREATED D of size"
        print sys.getsizeof(D)
    
        D = [tuple(y) for y in D]
        D = set(D)

        print "CREATED D of size(after set)"
        print sys.getsizeof(D)

        A = A.tolist()
        A = set([tuple(y) for y in A])


        all_pairs = [(x,y) for ((x,y),z) in np.ndenumerate(G)]
        all_pairs = set(all_pairs)
        B = list(all_pairs-A)[:4*ratio]
        # print "ALL PAIRS", len(all_pairs)
        # print "B PAIRS", len(B)

        A = None
        all_pairs = None

        gc.collect()

        print "CREATED B of size"
        print sys.getsizeof(B)
        

        B = [tuple(b) for b in B]
        B = set(B)

        print "CREATED B of size (after set)"
        print sys.getsizeof(B)

        # DUB = D.union(B)

        # ranking DUB pairs in decreasing order of confidence
        rank_dict_1 = {}
        for pair in itertools.chain(B, list(D)):
            (x,y) = pair
            rank_dict_1[(pair[0],pair[1])] = G_final[x-1,y-1]

        B = None

        gc.collect()
   
        rank_dict_1 = OrderedDict(sorted(rank_dict_1.items(), key=lambda kv: kv[1], reverse=True))

        rank_dict_1 = rank_dict_1.keys()
                
        E = set(rank_dict_1[:len(D)])

        rank_dict_1 = None

        gc.collect()

        TP = (float)(len(D.intersection(E)))/len(D)

        print "CALCULATED TP accuracy value"       
        
        D = None
        E = None

        return TP 

def calculateAccuracy(G_final,G,G_original):
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
      

calcTrust(U,H,G,G_original)

#NOTE================================
#if accuracy too low, switch with iterative solution to calculate B
# also calculate DUB instead of itertool