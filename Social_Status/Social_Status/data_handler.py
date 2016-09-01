import numpy as np
from scipy.io import loadmat
import collections
import math




class data_handler():

    def __init__(self,rating_path,trust_path):
        self.rating_path = rating_path
        self.trust_path = trust_path
        self.n = 0
        self.k = 0
        self.d = 0

    def get_stats(self):
        return self.n_users, self.n_prod, self.n_cat


        #BRING G BACK BEFORE RUNNING MAIN--------------------

    def load_matrices(self):
        #Loading matrices from data
        f1 = open(self.rating_path)
        f2 = open(self.trust_path)
        P_initial = loadmat(f1) #user-rating matrix
        G_raw = loadmat(f2) #trust-trust matrices
        #G_raw = np.array([])
        P_initial = P_initial['rating_with_timestamp']
        G_raw = G_raw['trust']
        print "SHAPE"
        print G_raw.shape

        self.n = max(P_initial[:,0]) 
        print "N: " + str(self.n)
        self.k = max(P_initial[:,1])  
        self.d = max(P_initial[:,2]) 
        P = np.zeros((self.n+1,self.k+1))


        for row in P_initial:
            i = row[0]
            k = row[1]
            P[i,k] = row[3]

        P_size = P.shape[0]

        #choosing 70% data for training and rest for testing

       
        G_needed = np.zeros((self.n+1,self.n+1))
        for row in G_raw:
            G_needed[row[0],row[1]] = 1

        test_value = self.n * 1
        print test_value
        G_needed = G_needed[:test_value]
        G_needed = np.array([x[:test_value] for x in G_needed])

        G_raw_refined =[]
        for pair in G_raw:
            if pair[0] < test_value and pair[1] < test_value:
                G_raw_refined.append(pair)
        G_raw_refined = np.array(G_raw_refined)

        print G_raw_refined


        #remove for actual run--------------

        # for (x,y) in np.ndenumerate(G_needed):
        #     [i,j] = [x[0],x[1]]
        #     if G_needed[i,j] == 1:
        #         G_raw.append([i+1,j+1])
        # G_raw = np.array(G_raw)
        

        #-----------------------------

        P = P[:test_value]
        # print "THIS IS TRUST MATRIX"
        # print G_needed
        # print len(G_needed)
        # print len(P)


        return [P, G_needed, self.d, G_raw_refined]

# data = data_handler("rating_with_timestamp.mat", "trust.mat")
# data.load_matrices()


#TODO - get P matrix from data (iXk matrix)






