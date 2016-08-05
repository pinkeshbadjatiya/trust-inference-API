from __future__ import print_function, division
import numpy as np
from numpy.linalg import norm
from sklearn import linear_model
from datetime import datetime
import sys, os, copy, sys, threading, time, pdb


sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__))))
from handler import data_handler

sys.path.append("./factorization")
from factorize_GRAD import mat_fact_GRAD
from factorize_AF import mat_fact_AF
from factorize_PYMF_SKLEARN import mat_fact_PYMF, mat_fact_SKLEARN

from globs import *
CURRENT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__))) + "/"
DATASET_NAME = CURRENT_DIR + DATASET_NAME


if not os.path.exists(FILE_DIR):
    os.makedirs(FILE_DIR)

from utils import log as logger
log = logger()

np.random.seed(RANDOM_SEED_FACTORIZATION)



class MATRI(object):

    def __init__(self, data):
        log.updateHEAD("Initializing MATRI...")
        self.max_iter = GLOBAL_max_itr
        self.t = GLOBAL_t
        self.r = GLOBAL_r
        self.l = GLOBAL_l
        self.T, self.mu, self.x, self.y, self.k, self.deleted_edges, self.node_to_index, self.rating_map  = data.load_data()


        # Compute deleted_k, i.e. all the ordered trust pairs in the test dataset
        # (u, v, r) :: (trustor(from), trustee(to), rating)
        self.deleted_k = []
        deleted_nodes = self.deleted_edges.keys()
        for i, node in enumerate(deleted_nodes):
            edge_list = self.deleted_edges[node]
            for user in edge_list:
                u = self.node_to_index[node]
                v = self.node_to_index[user]
                r = self.rating_map[edge_list[user]['level']]
                self.deleted_k.append((u,v,r))

        self.Z = np.zeros((len(self.k), 1, 4*self.t-1))

        self.__iter = 1

        if os.path.isfile(FILE_Z_train + ".npy"):
            log.updateHEAD("Loading Z from: %s.npy" %(FILE_Z_train))
            self.Z = np.load(FILE_Z_train + ".npy")
        else:
            log.updateHEAD("Precomputing Zij:")
            L, R = mat_fact_PYMF(self.T, self.l)
            # L, R, _d = mat_fact_AF(self.T, self.T.shape[0], self.l, self.k)
            #L, R = mat_fact_GRAD(T, l)

            _total = len(self.k)
            _estimated_time_left = " ------------- "
            _time = datetime.now()
            for ind, (i,j) in enumerate(self.k):
                log.updateMSG("Computing %d/%d Zij matrices.        || Estimated Time Left: %s" %(ind+1, len(self.k), _estimated_time_left))
                self.Z[ind] = data.compute_prop(L, R, self.t, self.l, i, j)

                # Just estimate the remaining time
                m, s = divmod((((datetime.now() - _time).total_seconds())*(_total - ind - 1)) / (ind + 1), 60)
                h, m = divmod(m, 60)
                _estimated_time_left = "%d hours %02d minutes %02d seconds" % (h, m, s)

            np.save(FILE_Z_train, self.Z)

        # Initializing the weight vectors
        self.alpha = np.array([1,1,1])
        self.beta = np.zeros((1, 4 * self.t - 1))

        self.F = np.zeros((data.num_nodes, self.l))
        self._oldF = np.zeros((data.num_nodes, self.l))
        self.G = np.zeros((data.num_nodes, self.l))
        self._oldG = np.zeros((data.num_nodes, self.l))

        #self.F = np.random.rand(data.num_nodes, self.l)
        #self._oldF = np.random.rand(data.num_nodes, self.l)
        #self.G = np.random.rand(data.num_nodes, self.l)
        #self._oldG = np.random.rand(data.num_nodes, self.l)
        #self._oldG = self.G = np.zeros((self.l, data.num_nodes))


    def updateCoeff(self, P):
        """ Update the Alpha and Beta weight vectors after each iteration
        """

        log.updateMSG("Updating the Alpha AND Beta weight vectors")
        b = np.zeros((len(self.k)))
        for ind, (i, j) in enumerate(self.k):
            b[ind] = P[i, j]

        A = np.zeros((len(self.k), 4*self.t + 2))
        A[:,0] = self.mu
        for ind,(i,j) in enumerate(self.k):
            A[ind,1] = self.x[i]
            A[ind,2] = self.y[j]
            # Dimension of Zt vs A[:,:,4:]
            # dim(Z[i,j]) = (1,4t-1)
            # dim(A[i,j]) = (1,4t+2)
            # and we skip the 1st 3 rows of A, therefore A's dimension available: (1,4t-1)
            A[ind,3:] = self.Z[ind,:,:]
        clf = linear_model.Ridge(alpha = 0.1)
        clf.fit(A, b)

        # The resultant vector is the concat. of the 2 vectors. Hence we need to split it into vectors,
        # Dimension:    Alpha=(1,3) ,  Beta=(1,4t-1)
        self.alpha, self.beta = np.split(clf.coef_, [3])    # Split the matrix into concat of Alpha and Beta



    def calc_successive_NORM(self):
        """ Calculate the successive NORM b/w self.F and self.G
        """
        log.updateMSG("Calculating NORM.")
        E1 = norm(self.F - self._oldF)
        E2 = norm(self.G - self._oldG)

        # Copy the successive F and G for next iteration.
        self._oldF = copy.deepcopy(self.F)
        self._oldG = copy.deepcopy(self.G)
        return E1, E2


    def is_converged(self, E1, E2, rmse):
        """ Returns True if Converged, else return False """

        # Skip for the 1st iteration
        if self.__iter is None:
            self.__iter = 1

        # Max iterations reached
        if self.__iter >= self.max_iter:
            return True

        # Update the iteration number
        self.__iter += 1

        log.updateMSG("MatrixConvergence: (%f, %f) || RMSE: %f" %(E1, E2, rmse))
        if E1 < GLOBAL_EPS and E2 < GLOBAL_EPS:
            return True
        return False


    def startMatri(self):
        """ Start the main MATRI algorithm """
        log.updateHEAD("Starting MATRI")

        RMSE = []

        # Compute Zij for test data
        self.Zij_test = self.compute_zij_test()

        # Join zij_test & zij_train
        self.Zij = self.join_zij()

        while True:
            P = np.zeros(self.T.shape)
            log.updateHEAD("Iteration: %d" % (self.__iter))
            for ind,(i,j) in enumerate(self.k):
                P[i, j] = self.T[i, j] - np.dot(self.alpha, np.asarray([self.mu, self.x[i], self.y[j]]).T) + \
                        np.dot(self.beta, self.Z[ind].T)

            self.F, self.G = mat_fact_PYMF(P, self.r)                          # So using pymf factorization
            # self.F, self.G, _d = mat_fact_AF(P, P.shape[0], self.r, self.k)
            #self.F, self.G = mat_fact_GRAD(P, self.r)

            for i,j in self.k:
                P[i, j] = self.T[i, j] - np.dot(self.F[i, :], self.G[j, :].T)

            # Update Alpha & Beta vectors
            self.updateCoeff(P)

            # Calculate convergence & RMSE
            _E1, _E2 = self.calc_successive_NORM()
            if not self.__iter % 1:
                self.calcTrust_test()
                _R = self.RMSE_test()
                RMSE.append(_R)
            else:
                _R = -1
            if self.is_converged(_E1, _E2, _R):
                break

        # Save all the RMSE to plot the graph
        np.save(FILE_RMSE, np.asarray(RMSE))
        log.nextLine()


    def join_zij(self):
        """ Computes the full Zij by joining all the values
            from the Test-Zij and Train-ij
        """
        Zij = np.zeros((data.num_nodes, data.num_nodes, 1, 4*self.t-1))
        if os.path.isfile(FILE_Z + ".npy"):
            log.updateHEAD("Loading FULL-Zij from: %s.npy" %(FILE_Z))
            Zij = np.load(FILE_Z + ".npy")
        else:
            # Copy Test-Zij
            for ind, (i,j,r) in enumerate(self.deleted_k):
                Zij[i,j] = self.Zij_test[i,j]

            # Copy Train-Zij
            for ind, (i,j) in enumerate(self.k):
                Zij[i,j] = self.Z[ind]

            # Grab whatever you can !! (Save to file)
            np.save(FILE_Z, Zij)
        return Zij


    def compute_zij_test(self):
        """ Computes the Zij on all the values
            from the test_dataset.
        """
        Zij = np.zeros((data.num_nodes, data.num_nodes, 1, 4*self.t-1))
        if os.path.isfile(FILE_Z_test + ".npy"):
            log.updateHEAD("Loading TEST-Zij from: %s.npy" % ( FILE_Z_test))
            Zij = np.load(FILE_Z_test + ".npy")
        else:
            L, R = mat_fact_PYMF(self.T, self.l)
            # L, R, _d = mat_fact_AF(self.T, self.T.shape[0], self.l, self.k)
            for ind, (i,j,r) in enumerate(self.deleted_k):
                log.updateMSG("Computing %dth Zij matrices. | TEST_DATASET" %(ind))
                Zij[i,j] = data.compute_prop(L, R, self.t, self.l, i, j)
                ind += 1
            np.save(FILE_Z_test, Zij)
        return Zij


    def compute_zij(self):
        """ Computes the FULL-Zij on all n^2 values
        """
        Zij = np.zeros((data.num_nodes, data.num_nodes, 1, 4*self.t-1))
        if os.path.isfile(FILE_Z + ".npy"):
            log.updateHEAD("Loading FULL-Zij from: %s.npy" %(FILE_Z))
            Zij = np.load(FILE_Z + ".npy")
        else:
            ind = 1
            total = data.num_nodes*data.num_nodes
            L, R = mat_fact_PYMF(self.T, self.l)
            # L, R, _d = mat_fact_AF(self.T, self.T.shape[0], self.l, self.k)
            for i in range(0, data.num_nodes):
                for j in range(0, data.num_nodes):
                    Zij[i,j] = data.compute_prop(L, R, self.t, self.l, i, j)
                    ind += 1
                    log.updateMSG("Computing %d/%d Zij matrices." %(ind, total))
            np.save(FILE_Z, Zij)
        return Zij



    def calcTrust_test(self):
        """ Calculate final trust values for (u,v) belongs to test-dataset """

        log.updateMSG("Calculating Trust values for TEST-dataset")
        self.Tnew = np.zeros((data.num_nodes, data.num_nodes))

        for ind, (u,v,r) in enumerate(self.deleted_k):
            # Used self.Z[u,v] instead of self.Z[u,v].T, as numpy gives transpose of the 1-d array as the array itself.
            # Use G[v,:] instead of transpose
            A = np.dot(self.F[u,:], self.G[v,:])
            B = np.dot(self.alpha.T, np.asarray([self.mu, self.x[u], self.y[v]]))
            C = np.dot(self.beta, self.Zij[u,v].T)
            self.Tnew[u,v] = A + B + C


# NOT USING FOR NOW!
#######################################
#
#    def calcTrust(self):
#        """ Calculate final trust values for all; (u,v) belongs T """
#        self.Tnew = np.zeros((data.num_nodes, data.num_nodes))
#
#        for u in range(0, data.num_nodes):
#            for v in range(0, data.num_nodes):
#                # Used self.Z[u,v] instead of self.Z[u,v].T, due to numpy issues
#                # Use G[v,:] instead of transpose
#                A = np.dot(self.F[u,:], self.G[v,:])
#                #pdb.set_trace()
#                B = np.dot(self.alpha.T, np.asarray([self.mu, self.x[u], self.y[v]]))
#                C = np.dot(self.beta, self.Zij[u,v].T)
#                self.Tnew[u,v] = A + B + C
#
#
#
#    def RMSE(self):
#        """ Calculate RMSE b/w the trust matrices
#        """
#        log.updateMSG("Calculating RMSE on the FULL-dataset.")
#        R = np.sqrt(np.mean((self.Tnew-self.T)**2))
#        log.updateMSG("RMSE (on FULL-dataset): %f" %(R))
#        return R
#
#######################################

    def RMSE_test(self):
        """ Calculate RMSE only on the test data, i.e 500 edges
        """
        log.updateMSG("Calculating RMSE on the TEST-dataset.")
        tvalue_test = np.array([])
        tvalue_train = np.array([])

        for ind, (n1,n2,r) in enumerate(self.deleted_k):
            tvalue_test = np.append(tvalue_test, r)
            tvalue_train = np.append(tvalue_train, self.Tnew[n1][n2])

        R = np.sqrt(np.mean(np.square(np.subtract(tvalue_test, tvalue_train))))
        log.updateMSG("RMSE (on TEST-dataset): %f" %(R))
        return R



def init_main():
    data = data_handler(DATASET_NAME, RATING_MAP, GLOBAL_t)
    m = MATRI(data)
    m.startMatri()

if __name__ == "__main__":
    init_main()
