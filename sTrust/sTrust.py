import numpy as np
import math
from data_handler_sTrust import data_handler
import operator
from operator import itemgetter
from collections import OrderedDict
import os


CURRENT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "./")
_rating_path = os.path.abspath(CURRENT_DIR + "data/rating_with_timestamp.mat")
_trust_path = os.path.abspath(CURRENT_DIR + "data/trust.mat")
_trust_with_timestamp_path = os.path.abspath(CURRENT_DIR + "data/epinion_trust_with_timestamp.mat")


class Social_Status:
    def __init__(self,G,P,d,max_itr, G_original):

        self.G_original = np.array(G_original,dtype = np.float32)
        self.G = G
        self.n = len(self.G)
        self.max_itr = max_itr
        self.d = d #number of user perferences (facets)
        self.alpha = 0.1
        # self.lambda1 = 0.1
        self.lambda2 = 0.7
        self.P = P #user-rating matrix (ixk)
        self.R = np.zeros((self.n,self.n),dtype = np.float32)
        self._oldU = np.zeros((self.n, self.d),dtype = np.float32)
        self.U= np.zeros((self.n, self.d),dtype = np.float32)
        self._oldH = np.zeros((self.d, self.d),dtype = np.float32)
        self.H = np.zeros((self.d, self.d),dtype = np.float32)
        self.G_final = np.zeros((self.n,self.n),dtype = np.float32)
        self.G_itr = np.zeros((self.n,self.n))
        self.Q = np.zeros((self.n,self.n),dtype = np.float32)
        self.TP = 0


    def pagerank(self,graph, damping=0.85, epsilon=1.0e-8):
        #ranks users in graph based on pagerank formulation

        inlink_map = {}
        outlink_counts = {}


        def new_node(node):
            if node not in inlink_map: inlink_map[node] = set()
            if node not in outlink_counts: outlink_counts[node] = 0

        for tail_node, head_node in graph:
            new_node(tail_node)
            new_node(head_node)
            if tail_node == head_node: continue

            if tail_node not in inlink_map[head_node]:
                inlink_map[head_node].add(tail_node)
                outlink_counts[tail_node] += 1

        all_nodes = set(inlink_map.keys())
        for node, outlink_count in outlink_counts.items():
            if outlink_count == 0:
                outlink_counts[node] = len(all_nodes)
                for l_node in all_nodes: inlink_map[l_node].add(node)

        initial_value = 1 / len(all_nodes)
        ranks = {}
        for node in inlink_map.keys(): ranks[node] = initial_value

        new_ranks = {}
        delta = 1.0
        n_iterations = 0
        while delta > epsilon:
            new_ranks = {}
            for node, inlinks in inlink_map.items():
                new_ranks[node] = ((1 - damping) / len(all_nodes)) + (damping * sum(ranks[inlink] / outlink_counts[inlink] for inlink in inlinks))
            delta = sum(abs(new_ranks[node] - ranks[node]) for node in new_ranks.keys())
            ranks, new_ranks = new_ranks, ranks
            n_iterations += 1

        print "FOUND PAGERANK RANKING"
        return ranks, n_iterations


    def determine_user_ranking(self):
        #creates user graph and uses pagerank to rank users
        user_directed_graph = {}
        for pair in self.G_original:
            [user1,user2] = pair
            if user1 not in user_directed_graph:
                user_directed_graph[user1] = [user2]
            else:
                user_directed_graph[user1].append(user2)


        graph = self.G_original.tolist()
        damping = 0.85
        epsilon = 1.0e-8
        rank_dict = self.pagerank(graph,damping,epsilon)[0]

        rank = [0] * (self.n + 1)

        #rank has format: rank[user] = ranking
        for user in rank_dict:
            user = int(user)
            rank[user] = rank_dict[user]

        rank = rank[1:]

        print "GOT USER RANKING"

        return np.asarray(rank)


    def calcR(self, test_1, final):

        # checks if trust(i,j) > trust(j,i)
        test_2 = self.G_itr - self.G_itr.T
        test_2 = test_2 > 0

        test = test_1 & test_2

        #setting where condition is False to 0, all others have final values
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
        EPS = 0.001
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

        term_1 = np.dot(np.dot(self.U,self.H),self.U.transpose()) #UHUt
        term_2 = np.dot(np.dot(self.U,self.H.transpose()),self.U.transpose()) #UHtUt

        A_1 = np.dot(np.dot(self.G.transpose(),self.U),self.H)
        A_2 = np.dot(np.dot(self.lambda2 * self.R * self.R * term_1,self.U),self.H.transpose())
        A_3 = np.dot(np.dot(self.lambda2 * self.R.transpose() * self.R.transpose() * term_2,self.U),self.H)
        A_4 = np.dot(np.dot(self.G,self.U),self.H.transpose())
        A_5 = np.dot(np.dot(self.lambda2 * self.R * self.R * term_2,self.U),self.H)
        A_6 = np.dot(np.dot(self.lambda2 * self.R.transpose() * self.R.transpose() * term_1,self.U),self.H.transpose())

        A = A_1 + A_2 + A_3 + A_4 + A_5 + A_6


        B_1 = np.dot(np.dot(term_2,self.U),self.H)
        B_2 = np.dot(np.dot(term_1,self.U),self.H.transpose())
        B_3 = self.alpha * self.U
        B_4 = np.dot(np.dot(2 * self.lambda2 * self.R.transpose() * self.R.transpose() * term_1, self.U),self.H)
        B_5 = np.dot(np.dot(2 * self.lambda2 * self.R * self.R * term_2, self.U),self.H.transpose())

        B = B_1 + B_2 + B_3 + B_4 + B_5


        C_1 = np.dot(np.dot(self.U.transpose(),self.G),self.U)
        C_2 = np.dot(np.dot(self.lambda2 * self.U.transpose(), (self.R * self.R * term_1)), self.U)
        C_3 = np.dot(np.dot(self.lambda2 * self.U.transpose(), (self.R.transpose() * self.R.transpose() * term_1)), self.U)
        C = C_1 + C_2 + C_3


        D_1 = np.dot(np.dot(self.U.transpose(),term_1),self.U)
        D_2 = self.alpha * self.H
        D_3 = np.dot(np.dot(2 * self.lambda2 * self.U.transpose(), (self.R * self.R * term_2)),self.U)

        D = D_1 + D_2 + D_3



        #updating self.U and self.H

        test_B = B != 0
        self.U = self.U * np.sqrt(A / B)
        indices_1 = zip(*np.where(test_B==False))

        # whenever divide by zero occurs, use old matrix value
        for x,y in indices_1:
            self.U[x,y] = self._oldU[x,y]


        test_D = D != 0
        self.H = self.H * np.sqrt(C / D)
        indices_2 = zip(*np.where(test_D==False))

        # whenever divide by zero occurs, use old matrix value
        for x,y in indices_2:
            self.H[x,y] = self._oldH[x,y]


        #desperate attempt to reduce mem usage

        A = None
        A_1 = None
        A_2 = None
        A_3 = None
        A_4 = None
        A_5 = None
        A_6 = None
        A_7 = None
        B = None
        B_1 = None
        B_2 = None
        B_3 = None
        B_4 = None
        B_5 = None
        C = None
        C_1 = None
        C_2 = None
        C_3 = None
        D = None
        D_1 = None
        D_2 = None
        D_3 = None
        indices_1 = None
        indices_2 = None
        test_B = None
        test_G = None


    def start_main(self):
        max_itr = 40

        ranking = self.determine_user_ranking()
        ranking = ranking.reshape(self.n,1)

        print "GOT FINAL RANK"

        #checks if rank(j) > rank(i)
        test_1 = (ranking.T - ranking) * -1
        test_1 = test_1 > 0


        print "DEVELOPED TEST_1"
        #calculates all the values
        final = (1/(1+(np.log(ranking-ranking.T +1))))


        print "FINAL VALUES"

        # self.U = np.random.random((self.n,self.d))
        # self.H = np.random.random((self.d,self.d))

        self.U = np.random.uniform(0.0,0.1,(self.n,self.d))
        # np.save("self_U",self.U)
        # self.U = np.load("self_U.npy")
        self.H = np.random.uniform(0.0,0.1,(self.d, self.d))
        # np.save("self_H",self.H)
        # self.H = np.load("self_H.npy")

        # self.U = np.ones((self.n,self.d)) * 0.1
        # self.H = np.ones((self.d,self.d)) * 0.1

        i = 1


        self.G_itr = np.dot(np.dot(self.U,self.H),self.U.transpose())
        # np.save("G_itr", self.G_itr)
        # self.G_itr = np.load("G_itr.npy")

        print "MADE FIRST G_ITR"

        print "Starting iterations......."

        # print self.converge(i)
        while not self.converge(i):
            print ("Iteration: ", i)

            self.calcR(test_1, final)

            self.updateMatrices()
            self.G_itr = np.dot(np.dot(self.U,self.H),self.U.transpose())

            i = i + 1

        print "Found U and H successfully!"
        print "THEY ARE"
        print self.U, self.H

        # np.save("self_U",self.U)
        # np.save("self_H",self.H)
        # np.save("self_G_original", self.G_original)
        # np.save("self_G",self.G)

        #cleaning up some variables
        ranking = None
        test_1 = None
        final = None
        self.G_itr = None
        self._oldH = None
        self._oldU = None
        self.Q = None

        self.calcTrust()


    def calcTrust(self):
        #calculate all final trust values knowing U,H

        self.G_final = np.dot(self.U,self.H)
        self.G_final = np.dot(self.G_final, self.U.transpose())

        # np.save("G_final_sTrust", self.G_final)

        print "Calculated final G. Doing accuracy things"

        print "Found predicted trust! Has TP accuracy: " + str(self.TP_accuracy())

        print "FINAL PREDICTED"
        print self.G_final

        print "START OFF AS"
        print self.G

        return self.G_final



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




def init_main():
    data = data_handler(_rating_path, _trust_path, _trust_with_timestamp_path)
    data = data.load_matrices()
    print "LOADED MATRICES"
    d = data[2]
    print "set d: " + str(d)
    P = data[0] # 0 for testing, 1 for training
    print "set p" +  str(len(P))
    G = data[1]
    print "set G" + str(len(G))
    G_original = data[3]
    obj = Social_Status(G,P,d,40,G_original)
    print "MADE SS OBJECT"
    obj.start_main()

    return


if __name__ == "__main__":
    init_main()



  #NOTE FOR RUN:
  # Cleaned out unimportant variables before they go out of scope
  # Without user filter

  #TODO
  #run on server with new changes
  # check mem things once again




