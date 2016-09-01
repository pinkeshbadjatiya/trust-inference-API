import numpy as np
import math
# from data_handler_new import data_handler
from trial_data import data_handler
import operator
from operator import itemgetter
from collections import OrderedDict
import pdb

#deleted self loops


class Social_Status:
    def __init__(self,G,P,d,max_itr, G_original):

        self.G_original = np.array(G_original)
        self.G = G
        self.n = len(self.G)
        self.max_itr = max_itr
        self.d = d #number of user perferences (facets)
        self.alpha = 0.1
        self.lambda1 = 0.1
        self.lambda2 = 0.1
        self.Z = np.zeros((self.n,self.n), dtype = np.float32)
        self.P = P #user-rating matrix (ixk)
        self.W = np.ones((self.n,self.n), dtype = np.float32) 
        self.R = np.zeros((self.n,self.n), dtype = np.float32) 
        self._oldU = np.zeros((self.n, self.d), dtype = np.float32)
        self.U= np.zeros((self.n, self.d), dtype = np.float64)
        self._oldH = np.zeros((self.d, self.d), dtype = np.float32)
        self.H = np.zeros((self.d, self.d), dtype = np.float32)
        self.G_final = np.zeros((self.n,self.n), dtype = np.float32)
        self.G_itr = np.zeros((self.n,self.n), dtype = np.float32)
        self.Q = np.zeros((self.n,self.n), dtype = np.float32)
        self.TP = -1

       

    def calcZ(self): 
        print "Starting Z calc"
        numerator = np.dot(self.P, self.P.transpose())
        print "made numerator"
        # J_norm = np.array([math.sqrt(x.sum()**2) for x in self.P]).reshape(self.n,1)
        J_norm = np.sum(np.abs(self.P)**2,axis=-1)**(1./2)

        J_norm = J_norm.reshape(self.n,1)
        print "made J_norm"
        I_norm = J_norm.transpose()

        denominator = np.dot(J_norm,I_norm)
        print "made denominator"

        # dividing numerator and denominator and handling nan and inf values

        with np.errstate(divide='ignore', invalid='ignore'):
            Z = np.divide(numerator,denominator)
            # Z[Z == np.inf] = 0
            Z = np.nan_to_num(Z)

        self.Z = Z
        print "FOUND Z!!"
    
    def calcW(self):
        # using np operations to construct boolean matrix

        Z_condition = (self.Z == 0)
        G_condition = (self.G == 0)
        total = Z_condition & G_condition

        total[np.where(total==True)] = 0.5
        total[np.where(total==False)] = 1
        print "FOUND W!"

        self.W = total
             

    def determine_user_ranking(self): #on basis of number of trustors
        users = np.arange(len(self.G)+1)[1:]
        
        # calculating number of trustors of each user
        trustor_number = {}
        for user in users:
            trustors = 0
            for i in range(0,len(users)):
                if self.G[i][user-1] != 0:
                    trustors = trustors + 1
            trustor_number[user] = trustors

        # ranking with highest number of trustors first
        sorted_x = sorted(trustor_number.items(), key=operator.itemgetter(1))
        rank = [x for (x,y) in sorted_x]

        # ranking in format: rank_list[user] = user_rank
        rank_final = np.zeros(self.n + 1)
        for user in rank:
            rank_final[user] = rank.index(user) + 1
        rank_final = rank_final[1:]
       
        return np.matrix(rank_final)

    def calcR(self, test_1,final):
        
        # checks if trust(i,j) > trust(j,i)
        test_2 = self.G_itr - self.G_itr.T 
        test_2 = test_2 > 0 
        
        test = test_1 & test_2

        final[np.where(test==False)] = 0
        final = np.sqrt(final)
        self.R = final
        print "FOUND R!"

    def converge(self, iterNO):
        #Returns True if Converged, else return False
        # Max iterations reached
        if iterNO >= self.max_itr:
            return True

        # Convergence is reached
        # EPS = np.finfo(float).eps
        EPS = 0.000001
        E1 = np.absolute(np.linalg.norm(self.U, ord = 'fro') - np.linalg.norm(self._oldU, ord = 'fro'))
        E2 = np.absolute(np.linalg.norm(self.H, ord = 'fro') - np.linalg.norm(self._oldH, ord = 'fro'))
        if E1 < EPS and E2 < EPS:
            if iterNO != 1:   # Skip for the 1st iteration
                print("\rIteration: %d FinalError: (%f, %f) EPS:%f" %(iterNO, E1, E2, EPS))
                return True

        self._oldU = self.U[:]
        self._oldH = self.H[:]
        
        print "E1 Difference: " + str(E1-EPS) + "E2 Difference: " + str(E2-EPS)

        #print("\rIteration: %d FinalError: (%f, %f) EPS:%f" %(iterNO, E1, E2, EPS))
        return False

    def updateMatrices(self):

        term_1 = np.dot(np.dot(self.U,self.H),self.U.transpose())
        term_2 = np.dot(np.dot(self.U,self.H.transpose()),self.U.transpose())

        A_1 = np.dot(np.dot((2 * self.W.transpose() * self.W.transpose() * self.G.transpose()),self.U),self.H)
        A_2 = np.dot(np.dot((2 * self.W * self.W * self.G),self.U),self.H.transpose())
        A_3 = 4 * self.lambda1 * np.dot(self.Z,self.U)
        A_4 = 2 * self.lambda2 * np.dot(np.dot((self.R.transpose() * self.R.transpose() * term_1),self.U),self.H.transpose())
        A_5 = 2 * self.lambda2 * np.dot(np.dot((self.R.transpose() * self.R.transpose() * term_2),self.U),self.H)
        A_6 = 2 * self.lambda2 * np.dot(np.dot((self.R  * self.R * term_2),self.U),self.H)
        A_7 = 2 * self.lambda2 * np.dot(np.dot((self.R  * self.R * term_1),self.U),self.H.transpose())
        A = A_1 + A_2 + A_3 + A_4 + A_5 + A_6 + A_7
        # print "HERE IS A"
        # print A

        B_1 = np.dot(np.dot((self.W.transpose() * self.W.transpose() * term_2),self.U),self.H)
        B_2 = np.dot(np.dot((self.W * self.W * term_1),self.U),self.H.transpose())
        B_3 = 2 * self.alpha * self.U
        B_4 = np.dot(np.dot((self.W.transpose() * self.W.transpose() * term_1),self.U),self.H.transpose())
        B_5 = np.dot(np.dot((self.W * self.W * term_2),self.U),self.H)
        B_6 = 4 * self.lambda1 * np.dot(self.Q,self.U)
        B_7 = 4 * self.lambda2 * (np.dot(np.dot((self.R.transpose() * self.R.transpose() * term_1),self.U),self.H))
        B_8 = 4 * self.lambda2 * np.dot(np.dot((self.R * self.R * term_2),self.U),self.H.transpose())
        B = B_1 + B_2 + B_3 + B_4 + B_5 + B_6 + B_7 + B_8
        # print "HERE IS B"
        # print B

        C_1 = 2 * self.lambda2 * np.dot(np.dot(self.U.transpose(),(self.R * self.R * term_1)),self.U)
        C_2 = 2 * self.lambda2 * np.dot(np.dot(self.U.transpose(),(self.R.transpose() * self.R.transpose() * term_1)),self.U)
        C_3 = 2 * np.dot(np.dot(self.U.transpose(),(self.W * self.W * self.G)),self.U)
        C = C_1 + C_2 + C_3
        # print "HERE IS C"
        # print C

        D_1 = np.dot(np.dot(self.U.transpose(),(self.W * self.W * term_1)),self.U)
        D_2 = np.dot(np.dot(self.U.transpose(),(self.W.transpose() * self.W.transpose() * term_1)), self.U)
        D_3 = 4 * self.lambda2 * np.dot(np.dot(self.U.transpose(),(self.R * self.R * term_2)), self.U)
        D_4 = 2 * self.alpha * self.H
        D = D_1 + D_2 + D_3 + D_4
        # print "HERE IS D"
        # print D

        # print "STARTING U,H UPDATES"

        # test_B = B != 0
        # self.U = self.U * np.sqrt(A / B)
        # self.U[np.where(test_B==False)] = 0

        # test_D = D != 0
        # self.H = self.H * np.sqrt(C / D)
        # self.H[np.where(test_D==False)] = 0


        test_B = B != 0
        # print self.U.shape
        # print A.shape, B.shape
       
        self.U = self.U * np.asarray(np.sqrt(A / B))
        indices_1 = zip(*np.where(test_B==False))
        # print "ORIGINAL LIST"
        # print np.where(test_B==False)
        for x,y in indices_1:
            self.U[x,y] = self._oldU[x,y]

        # self.U[np.where(test_B==False)] = self._oldU[np.where(test_B==False)]


        test_D = D != 0
        self.H = self.H * np.asarray(np.sqrt(C / D))
        indices_2 = zip(*np.where(test_D==False))
        for x,y in indices_2:
            self.H[x,y] = self._oldH[x,y]

        # print self._oldH[np.where(test_B==False)]

        # self.H[np.where(test_D==False)] = self._oldH[np.where(test_B==False)]


    def start_main(self):
        max_itr = 50
        self.calcZ()
        self.calcW()

        P = np.zeros(self.G.shape)
        L = np.zeros(self.G.shape)

        #calculating homophily contribution
        for i in range(0,self.n):
            total = 0
            for j in range(0,self.n):
                total = total + self.Z[j,i]
            self.Q[i,i] = total

        L = self.Q - self.Z

        ranking = self.determine_user_ranking()

        # checks if rank(j) > rank(i)
        test_1 = (ranking.T - ranking) * -1.
        test_1 = test_1 > 0

        #calculating all R values
        # final = (1/(1+np.log(ranking +1))) - (1/(1+np.log(ranking.T + 1)))
        final = (1/(1+np.log(ranking-ranking.T+1)))
        print "FINAL"
        print final

        #SEE WHY TURNS TO NAN-----------------

        self.U = np.random.random((self.n,self.d))
        self.H = np.random.random((self.d,self.d))
        self.G_itr = np.dot(np.dot(self.U,self.H),self.U.transpose())
        # self.U = np.ones((self.n,self.d)) * 0.1
        # self.H = np.ones((self.d,self.d)) * 0.1

        i = 1

    
        # print self.converge(i)
        while i < max_itr:
            print ("Iteration: ", i)
            self.calcR(test_1,final)
            # print "R"
            # print self.R

    
            self.updateMatrices()
            self.G_itr = np.dot(np.dot(self.U,self.H),self.U.transpose())
            #print self.U, self._oldU
            i = i + 1
        
        print "Found U and H successfully!"
        print "THEY ARE"
        print self.U, self.H
        self.calcTrust()

            

    def calcTrust(self):
        #calculate all final trust values knowing U,H

        self.G_final = np.dot(self.U,self.H)
        self.G_final = np.dot(self.G_final, self.U.transpose())

        print "Found predicted trust! Has TP accuracy: " + str(self.TP_accuracy())
        #print G_final

        # print "FINAL PREDICTED"
        # print self.G_final
        
        # print "START OFF AS"
        # print self.G
        np.save("G_final_hsTrust", self.G_final)
        return self.G_final


    def RMSE(self):
        return np.sqrt(np.mean((self.G_final - self.G)**2))

    def TP_accuracy(self):
        # set ratio of data to split
        q = 0.5
        # r,c = np.where(G > 0)
        # n_N = int(len(zip(r,c)) *0.5)
        # N = zip(r,c)[n_N:]
        # print "N: ", len(N)

        A = self.G_original
    
        n_N = int(len(self.G_original) *0.5) #getting middle number
        print n_N
        N = self.G_original[n_N:] #getting N as np array
        N = [tuple(x) for x in N] #turning N to a list of tuples
        
        print "N: ", len(N)
        # print N


        # N = random.sample(zip(r,c), n_N)
        # r,c = np.where(self.G == 0)
        # # B = random.sample(zip(r,c), n_N)
        # B = zip(r,c)
        # print "B: ", len(B)

        A = set([tuple(x) for x in A]) #turning A into a set
        all_pairs = [(a,b) for ((a,b),z) in np.ndenumerate(self.G)]
        all_pairs = set(all_pairs)
        B = list(all_pairs-A) 
        print "B PAIRS", len(B)

        print type(B), type(N)

        BUN = B + N
        T1 = []

        print "Union: ", len(BUN)
        
        for u, v in BUN:
            ptrust = self.G_final[u-1,v-1]
            T1.append((ptrust, u, v))
           
        T1.sort()
        T1.reverse()
       
        cnt1 = 0
        
        for i in xrange(n_N):
            if (T1[i][1],T1[i][2]) in N:
                cnt1 += 1
           
        print "CNT: ", cnt1
        print n_N
        self.TP = float(cnt1) / len(N)
        print PA1
       
        del BUN
        del N
        del B
        del T1
        
        print "X = ", q, "Prediction Accuracy (Betweenness nx)= ", self.TP
        return self.TP
    
    
data = data_handler("rating_with_timestamp.mat", "trust.mat", "epinion_trust_with_timestamp.mat")
data = data.load_matrices()
print "LOADED MATRICES"
d = data[2]
print "set d: " + str(d)
P = data[0] # 0 for testing, 1 for training 
print "set p" +  str(len(P))
G = data[1]
print "set G" + str(len(G))
G_original = data[3]
obj = Social_Status(G,P,d,50,G_original)
print "MADE SS OBJECT"
obj.start_main()

    


            
