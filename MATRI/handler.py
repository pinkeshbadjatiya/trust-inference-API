from __future__ import division
import networkx as nx
import json
import numpy as np
import pdb
from numpy import power
from datetime import datetime
import os
import copy
import sys

import io
from scipy.sparse import coo_matrix
from scipy.io import savemat

CURRENT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(CURRENT_DIR + "/dataset")

sys.path.append(CURRENT_DIR + "/factorization")
print(sys.path)
from factorize_PYMF_SKLEARN import mat_fact_PYMF,  mat_fact_SKLEARN
from factorize_AF import mat_fact_AF
from factorize_GRAD import mat_fact_GRAD


class data_handler(object):
    def __init__(self, path, rating_map, t):
        self.path = path
        self.rating_map = rating_map
        self.t = t

    def a2v(self, x):
        # Converts array of size (n,) to row vector - (1,n)
        return x.reshape(1, x.size)

    def compute_prop(self, L, R, t, l, i, j):
        # Pre-compute L & R and pass it here
        zij = []

        RtL = np.dot(R.T, L)
        for step in xrange(2, t+1):
            temp = power(RtL, step-1)
            zij.append(np.dot(np.dot(self.a2v(L[i,:]), temp), self.a2v(R[j,:]).T))

        LtR = np.dot(L.T, R)
        for step in xrange(1, t+1):
            temp = power(LtR, step-1)
            zij.append(np.dot(np.dot(self.a2v(R[i,:]), temp), self.a2v(L[j,:]).T))

        LtL = np.dot(L.T, L)
        RtR = np.dot(R.T, R)
        for step in xrange(1, t+1):
            temp = power(np.dot(LtL, RtR), step-1)
            zij.append(np.dot(np.dot(self.a2v(R[i,:]), temp), np.dot(LtL, self.a2v(R[j, :]).T)))
        for step in xrange(1, t+1):
            temp = power(np.dot(RtR, LtL), step-1)
            zij.append(np.dot(np.dot(self.a2v(L[i,:]), temp), np.dot(RtR, self.a2v(L[j, :]).T)))

        return np.asarray(zij).reshape(1, 4*t -1)

    def parse_dot(self, path):
        rating_map = self.rating_map
        f = io.open(path)
        usercount, user2idx = {}, {}
        userf, usert, rating = [], [], []
        no_users = 0

        for line in f:
            if line.find("->") != -1 and line.find("/*") == -1:
                useri, edge_info = line.split("->")
                useri = useri.strip()
                edge_info = edge_info.strip()
                userj, level_info = edge_info.split(" ")
                userj = userj.strip()
                level_info = level_info.strip()
                _, level = level_info.split("=")

                level = level[:-2]

                # Remove self-loops
                if useri == userj:
                    continue

                if useri not in user2idx:
                    user2idx[useri] = no_users
                    no_users += 1

                if userj not in user2idx:
                    user2idx[userj] = no_users
                    no_users += 1

                if useri in usercount:
                    usercount[useri] += 1
                else:
                    usercount[useri] = 1

                if userj in usercount:
                    usercount[userj] += 1
                else:
                    usercount[userj] = 1
                userf.append(useri)
                usert.append(userj)
                rating.append(level)

        row = map(lambda x: user2idx[x], userf)
        col = map(lambda x: user2idx[x], usert)
        rating = map(lambda x: rating_map[x], rating)
        idx2user = {v: k for k, v in user2idx.items()}

        # Count nodes
        nodes = row + col
        nodes = list(set([idx2user[i] for i in nodes]))

        edges_new = {}
        rev_rating_map = {v: k for k, v in rating_map.items()}
        for i in range(len(row)):
            u = idx2user[row[i]]
            v = idx2user[col[i]]
            r = rev_rating_map[rating[i]]
            if u not in edges_new:
                edges_new[u] = {}
            edges_new[u][v] = {'level': r}

        return nodes, edges_new

    def load_data(self):

        # graph = nx.DiGraph(nx.drawing.nx_pydot.read_dot(self.path))
        # nodes = graph.node.keys()
        # edges = copy.deepcopy(graph.edge)

        nodes, edges = self.parse_dot(self.path)

        REDUCE_DATA = True
        KEEP_EDGES = 500
        deleted_edges = {}
        if REDUCE_DATA:
            fileTrain = "data_train.txt"
            fileTest = "data_test.txt"
            if os.path.isfile(fileTrain) and os.path.isfile(fileTest):
                print("Loading the SPLIT dataset from file...")
                with open(fileTrain, 'r') as f:
                    edges = json.load(f)
                with open(fileTest, 'r') as f:
                    deleted_edges = json.load(f)
            else:
                print("Splitting dataset, Saving %d edges for testing" %(KEEP_EDGES))
                ind = 0
                while ind < KEEP_EDGES:
                    # Find a node
                    keys = edges.keys()
                    i = np.random.randint(0, len(keys))
                    n1 = keys[i]

                    # Find neighbour
                    neighbours = edges[n1]
                    keys2 = neighbours.keys()

                    # When the node has no neighbours then skip
                    if len(keys2) == 0:
                        continue
                    ind = ind + 1
                    j = np.random.randint(0, len(keys2))
                    n2 = keys2[j]

                    # Delete 'em
                    if n1 not in deleted_edges:
                        deleted_edges[n1] = {}
                    if n2 not in deleted_edges[n1]:
                        deleted_edges[n1][n2] = {}

                    deleted_edges[n1][n2].update(edges[n1].pop(n2))

                print("Saving dataset to files...")
                # Save only when we split the dataset
                with open('data_train.txt', 'w') as f:
                    json.dump(edges, f)
                with open('data_test.txt', 'w') as f:
                    json.dump(deleted_edges, f)

            deleted_num_edges = sum(map(lambda x:len(deleted_edges[x].keys()), deleted_edges))
            print('Deleted Edges = %d' %(deleted_num_edges))

        self.num_nodes = len(nodes)
        self.num_edges = sum(map(lambda x: len(edges[x].keys()), edges))
        print "Nodes:", self.num_nodes, ", Edges:", self.num_edges
        node_to_index = dict(zip(nodes, range(len(nodes))))

        T = np.zeros((self.num_nodes, self.num_nodes))
        k = []
        for i, node in enumerate(nodes):
            if node not in edges:
                continue
            edge_list = edges[node]
            for user in edge_list:
                T[node_to_index[node]][node_to_index[user]] = self.rating_map[edge_list[user]['level']]
                k.append((node_to_index[node], node_to_index[user]))

        mu = np.sum(T)
        mu /= len(T[np.where(T > 0)])
        x = np.zeros(self.num_nodes)
        y = np.zeros(self.num_nodes)
        for ind, (i, j) in enumerate(k):
            x[i] = np.sum(T[i, :]) / len(np.where(T[i,:] > 0))
            x[i] -= mu
            y[j] = np.sum(T[:, j]) / len(np.where(T[:, j] > 0))
            y[j] -= mu
        return T, mu, x, y, k, deleted_edges, node_to_index, self.rating_map

if __name__ == "__main__":
    rating_map = {'"Observer"':0.1, '"Apprentice"':0.4, '"Journeyer"':0.7, '"Master"':0.9}
    data = data_handler("dataset/advogato-graph-2000-02-25.dot", rating_map, 5)
    t = datetime.now()
    T, mu, x, y, k = data.load_data()
    t = (datetime.now() - t).total_seconds()
    print "Time for pre-processing is %fs"%(t)
