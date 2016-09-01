import numpy as np
import math
from data_handler import data_handler
import operator
from operator import itemgetter

class Social_Status:
    def __init__(self,G,P,d,max_itr, G_original):

        self.G_original = G_original
        self.G = G
        self.n = len(self.G)
        self.max_itr = max_itr
        self.d = d #number of user perferences (facets)
        self.alpha = 0.1
        self.lambda1 = 0.1
        self.lambda2 = 0.1
        self.Z = np.zeros((self.n,self.n)) 
        self.P = P #user-rating matrix (ixk)
        self.W = np.ones((self.n,self.n)) 
        self.R = np.zeros((self.n,self.n)) 
        self._oldU = np.zeros((self.n, self.d))
        self.U= np.random.random((self.n, self.d))
        self._oldH = np.zeros((self.d, self.d))
        self.H = np.random.random((self.d, self.d))
        self.G_final = np.zeros((self.n,self.n))
        self.Q = np.zeros((self.n,self.n))

        # self._oldU = np.zeros((5, 3))
        # self.U = np.random.random((5, 3))
        # self._oldH = np.zeros((3, 3))
        # self.H = np.random.random((3, 3))
        # self.max_iter = 150
        # self.W = np.random.random()
        # self.lambda1 = 0.1
        # self.lambda2 = 0.1
        # self.alpha = 0.1
        # self.d = 3
        # self.n = 5
        # self.G = np.random.randint(2, size=(5,5))
        # self.G_final = np.random.random(self.G.shape)

        # self.G_original = []
        # for (x,y) in np.ndenumerate(self.G):
        #     [i,j] = [x[0],x[1]]
        #     if self.G[i,j] == 1:
        #         self.G_original.append([i+1,j+1])
        # self.G_original = np.array(self.G_original)
        # # print self.G
        # #print self.G_final

        # self.Z = np.random.random(self.G.shape)
        # self.W = np.random.random((5,5))
        # self.R = np.random.random((5,5))
        # self.Q = np.diag(np.random.random((5,5)))
        # self.Q = np.diag(self.Q)
        # self.P = np.random.random((5,9))
        # self.known_user_set = set()


        # self.G = np.ma.masked_equal(self.G,0)


    
    #handling negative numbers?

    def calcZ(self): #faster?

        numerator = np.dot(self.P, self.P.transpose())
        I_norm = np.array([math.sqrt(sum(x)**2) for x in self.P]).reshape(self.n,1)

        J_norm = I_norm.transpose()

        denominator = np.dot(I_norm,J_norm)

        with np.errstate(divide='ignore', invalid='ignore'):
            Z = np.divide(numerator,denominator)
            Z[Z == np.inf] = 0
            Z = np.nan_to_num(Z)

        self.Z = Z
        print "FOUND Z!!"

        return Z
    
    def calcW(self):
        Z_condition = (self.Z != 0).astype(np.float32)
        G_condition = (self.G != 0).astype(np.float32)
        print "Z"
        print Z_condition
        print "G"
        print G_condition
        total = Z_condition + G_condition
        total[np.where(total==1)] = 0.2
        total[np.where(total==0)] = 1

        self.W = total
             

    def determine_user_ranking(self):
        users = np.arange(len(self.G)+1)[1:]
        print "USERS"
        print users
        print self.G
        trustor_number = {}
        for user in users:
            trustors = 0
            for i in range(0,len(users)):
                if self.G[i][user-1] != 0:
                    trustors = trustors + 1
            trustor_number[user] = trustors
        print "TRUSTOR DICT"
        print trustor_number
        sorted_x = sorted(trustor_number.items(), key=operator.itemgetter(1))

        rank = [x for (x,y) in sorted_x]
        # rank.reverse()
        print rank
        return rank


    def calcR (self):
        ranking = self.determine_user_ranking()
        print ranking

        for (x,y) in np.ndenumerate(self.Z):
            print (x,y) 
            i = x[0] 
            j = x[1] 

            if ranking.index(j+1) > ranking.index(i+1) and self.G[i,j] > self.G[j,i]: #for user i+1, j+1
                # s[i]-s[j] calculation
                # print "ERROR THINGS"
                # print (i+1,j+1)
                # print (ranking.index(i+1),ranking.index(j+1))
                # print (1/(1+math.log(ranking.index(i+1)+2)))-(1/(1+math.log(ranking.index(j+1)+1)))
                value = math.sqrt((1/(1+math.log(ranking.index(i+1)+2)))-(1/(1+math.log(ranking.index(j+1)+2))))
                self.R[i,j] = value
            
        print "Calculated R"
        print self.R

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

        self._oldU = np.copy(self.U)
        self._oldH = np.copy(self.H)
        

        #print("\rIteration: %d FinalError: (%f, %f) EPS:%f" %(iterNO, E1, E2, EPS))
        return False

    def updateMatrices(self):

        # print "LEN OF W" + str(self.W.shape)
        # print "LEN OF G" + str(self.G.shape)
        # print "LEN OF U" + str(self.U.shape)
        # print "LEN OF H" + str(self.H.shape)

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
        print "HERE IS A"
        print A

        B_1 = np.dot(np.dot((self.W.transpose() * self.W.transpose() * term_2),self.U),self.H)
        B_2 = np.dot(np.dot((self.W * self.W * term_1),self.U),self.H.transpose())
        B_3 = 2 * self.alpha * self.U
        B_4 = np.dot(np.dot((self.W.transpose() * self.W.transpose() * term_1),self.U),self.H.transpose())
        B_5 = np.dot(np.dot((self.W * self.W * term_2),self.U),self.H)
        B_6 = 4 * self.lambda1 * np.dot(self.Q,self.U)
        B_7 = 4 * self.lambda2 * (np.dot(np.dot((self.R.transpose() * self.R.transpose() * term_1),self.U),self.H))
        B_8 = 4 * self.lambda2 * np.dot(np.dot((self.R * self.R * term_2),self.U),self.H.transpose())
        B = B_1 + B_2 + B_3 + B_4 + B_5 + B_6 + B_7 + B_8
        print "HERE IS B"
        print B

        C_1 = 2 * self.lambda2 * np.dot(np.dot(self.U.transpose(),(self.R * self.R * term_1)),self.U)
        C_2 = 2 * self.lambda2 * np.dot(np.dot(self.U.transpose(),(self.R.transpose() * self.R.transpose() * term_1)),self.U)
        C_3 = 2 * np.dot(np.dot(self.U.transpose(),(self.W * self.W * self.G)),self.U)
        C = C_1 + C_2 + C_3
        print "HERE IS C"
        print C

        D_1 = np.dot(np.dot(self.U.transpose(),(self.W * self.W * term_1)),self.U)
        D_2 = np.dot(np.dot(self.U.transpose(),(self.W.transpose() * self.W.transpose() * term_1)), self.U)
        D_3 = 4 * self.lambda2 * np.dot(np.dot(self.U.transpose(),(self.R * self.R * term_2)), self.U)
        D_4 = 2 * self.alpha * self.H
        D = D_1 + D_2 + D_3 + D_4
        print "HERE IS D"
        print D

        for u in range(0,self.n):
            for v in range(0,self.d):
                if B[u,v] == 0:
                    self.U[u,v] = 0
                else:
                    self.U[u,v] = self.U[u,v] * math.sqrt(A[u,v]/B[u,v])

        # print self.U, self._oldU
        for u in range(0,self.d):
            for v in range(0,self.d):
                if D[u,v] == 0: #to adjust for zero division 
                    self.H[u,v] = 0
                else:
                    self.H[u,v] = self.H[u,v] * math.sqrt(C[u,v]/D[u,v])
                
        #print self.H, self._oldH

    def start_main(self):
        self.calcZ()
        self.calcW()
        self.calcR()

        P = np.zeros(self.G.shape)
        Q = np.zeros(self.G.shape)
        self.Q = Q
        L = np.zeros(self.G.shape)

        #calculating homophily contribution
        for i in range(0,self.n):
            total = 0
            for j in range(0,len(self.G)):
                total = total + self.Z[j,i]
            Q[i,i] = total

        L = Q - self.Z
        homo_contribution = 2 * np.trace(np.dot(np.dot(self.U.transpose(),L),self.U))
        print "Calculated homphily contribution"

        # #calculating status contribution
        part_2 = np.dot(np.dot(self.U,self.H.transpose()),self.U.transpose())
        part_3 = np.dot(np.dot(self.U,self.H),self.U.transpose())
        status_contribution = np.linalg.norm(self.R * (part_2 - part_3),ord = 'fro')
        status_contribution = status_contribution ** 2
        print "Calculated status contribution"

        self.U = np.random.random((len(self.G),self.d))
        self.H = np.random.random((self.d,self.d))

        i = 1

    
        print self.converge(i)
        while not self.converge(i):
            print ("Iteration: ", i)
            # self.calcR()
        #print self.U, self._oldU
        #term1 = np.linalg.norm(self.W * (self.G - np.dot(np.dot(self.U,self.H),self.U.transpose())),ord='fro')
        #term1 = term1**2
        #skipped the regulating terms as in MATRI/ frobenius norms?
        #term2 = self.lambda1 * homo_contribution
        #term3 = self.lambda2 * status_contribution

        #P = term1 + term2 + term3 #what was included in P?

        # self.updateMatrices()
            #print "AFTER UPDATE"
            self.updateMatrices()
            #print self.U, self._oldU
            i = i + 1
        
        print "Found U and H successfully!"
        print "THEY ARE"
        print self.U, self.H
        self.calcTrust()

            

    def calcTrust(self):
        #calculate all final trust values knowing U,H
        G_final = np.zeros((len(self.G),len(self.G)))

        G_final = np.dot(self.U,self.H)
        G_final = np.dot(G_final, self.U.transpose())

        self.G_final = G_final
        print "Found predicted trust! Has TP accuracy: " + str(self.TP_accuracy())
        #print G_final

        print "FINAL PREDICTED"
        print G_final
        
        print "START OFF AS"
        print self.G



        return G_final


    def RMSE(self):
        return np.sqrt(np.mean((self.G_final - self.G)**2))

    def TP_accuracy(self):
        ratio = int(len(self.G_original) *0.5)
        A = self.G_original.tolist()
        print "A"
        print type(A)
    
        C = A[:ratio]
        print C
        D = A[ratio:]
        print D
        print "DDDDDDDDDD"
        print len(D)
        D = [tuple(y) for y in D]
        D = set(D)

        print "Here's D"
        print D

        #memoization of B

        # val = values(x)

        # if val.B() != None:
        #     B = val.B()

        # else:

        B = [] #set of no trust user pairs

        for (x,y) in np.ndenumerate(self.G):
            if len(B) == len(D):
                break
            [i,j] = [x[0],x[1]]
            array = [i+1,j+1]
            if self.G[i,j] == 0:
                B.append(array)
                print "Done: " + str(len(B)) + " of " + str(len(A)- ratio)
        print "B IS THIS BIG" + str(len(B))

        B = [tuple(b) for b in B]
        B = set(B)
        DNB = D.union(B)

        rank_dict_1 = {}
        for pair in DNB:
            (x,y) = pair
            print pair
            rank_dict_1[(pair[0],pair[1])] = self.G_final[x-1,y-1]

        #rank_dict_1 = dict(sorted(rank_dict_1.items(), key=itemgetter(1), reverse = False))
        
        print "AFTER SORTING"
        print rank_dict_1
        rank_list = rank_dict_1.keys()
        rank_list.reverse()
        E = set(rank_list[:len(D)])
        print E

        TP = (float)(len(D.intersection(E)))/len(D)

        print TP

        return TP 
    
    
data = data_handler("rating_with_timestamp.mat", "trust.mat")
data = data.load_matrices()
print "LOADED MATRICES"
d = data[2]
print "set d: " + str(d)
P = data[0] # 0 for testing, 1 for training 
print "set p" +  str(len(P))
G = data[1]
print "set G" + str(len(G))
G_original = data[3]
obj = Social_Status(G,P,d,150,G_original)
print "MADE SS OBJECT"
obj.calcW()

    


            
