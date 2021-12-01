import os

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import pymetis
import numpy as np
import random
import statistics
import collections
import pathlib
import json
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from collections import defaultdict
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
import time

# def plotGraph(graph, pos):
#     nx.draw_networkx_edges(graph, pos)
#     nx.draw_networkx_nodes(graph, pos)
#     nx.draw_networkx_labels(graph, pos)
#     plt.axis("off")
#     plt.savefig("initial_karate_club_graph.png")

# -----------------------------------------------Facebook Large Page Page----------------------------------------
# # # Step1: Plot graph
# edges_path = 'facebook_large/musae_facebook_edges.csv'
# edges = pd.read_csv(edges_path)
# edges.columns = ['source', 'target']
#
# features_path = 'facebook_large/musae_facebook_features.json'
# with open(features_path) as json_data:
#     features = json.load(json_data)
#
# max_feature = np.max([v for v_list in features.values() for v in v_list])
# features_matrix = np.zeros(shape=(len(list(features.keys())), max_feature + 1))
#
# i = 0
# for k, vs in tqdm(features.items()):
#     for v in vs:
#         features_matrix[i, v] = 1
#     i += 1
#
# node_features = pd.DataFrame(features_matrix, index=features.keys())
# G = nx.from_pandas_edgelist(edges)
#
# print(edges.sample(frac=1).head(5))
# print("Number of nodes:", G.number_of_nodes())
# print("Number of edges:", G.number_of_edges())
# print("node feature", node_features)
# -------------------------------------------------Karate Club-------------------------------------------------------
######################################
# G = nx.karate_club_graph()
######################################
# pos = nx.spring_layout(G)
# plotGraph(G, pos)
# --------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------GIT-ML Dataset----------------------------------------------------
edges_path = 'git_web_ml/git_edges.csv'
targets_path = 'git_web_ml/git_target.csv'
features_path = 'git_web_ml/git_features.json'

# Read in edges
edges = pd.read_csv(edges_path)
edges.columns = ['source', 'target']  # renaming for StellarGraph compatibility

with open(features_path) as json_data:
    features = json.load(json_data)

max_feature = np.max([v for v_list in features.values() for v in v_list])
features_matrix = np.zeros(shape=(len(list(features.keys())), max_feature + 1))

i = 0
for k, vs in tqdm(features.items()):
    for v in vs:
        features_matrix[i, v] = 1
    i += 1

node_features = pd.DataFrame(features_matrix, index=features.keys())

print("number of features", node_features)
# Read in targets
targets = pd.read_csv(targets_path)
targets.index = targets.id.astype(str)
targets = targets.loc[features.keys(), :]

# Put the nodes, edges, and features into stellargraph structure
# G = sg.StellarGraph(node_features, edges.astype(str))     ; Comment because of stellargraph

edges = pd.read_csv(edges_path)
edges.columns = ['source', 'target']
print(edges.sample(frac=1).head(5))

G = nx.from_pandas_edgelist(edges)

print(edges.sample(frac=1).head(5))
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())
# --------------------------------------------------------------------------------------------------------------------

# Sampling Length
K = 5


# print("Number of nodes:", G.number_of_nodes())
# print("Number of edges:", G.number_of_edges())


def findDegree(graph):
    return [val for (n, val) in graph.degree()]
    # return [val for (node, val) in graph.degree()]


# Step2: Find degree of nodes
degrees = findDegree(G)


# print('Degrees of Each Node is:-', degrees)


def weight(node, sampling_length, degrees):
    probability_weight = 1 - pow((1 - 1 / degrees[node]), sampling_length)
    return int(probability_weight * 100)


eweights = []
adjacency_list = dict(G.adjacency())
adj = {}
adjncy = []
xadj = [0]
size = 1

# print("adjacency_list", adjacency_list)
for node in adjacency_list:
    adjacency = list(adjacency_list[node].keys())
    adj[node] = adjacency
    adjncy += adjacency
    # print("adjacency is ", adjacency)
    # Testing
    for neighbor in adjacency:
        # print("neighbors is", neighbor)             # Testing
        eweights.append(weight(neighbor, K, degrees))
        # print("eweight array is", eweights)         # Testing
    xadj.append(xadj[size - 1] + len(adjacency))
    size += 1

# print('adjacency list is:-', adjacency_list)
# print("adjncy Array is: ", adjncy)
# print("xadj Array is: ", xadj)
# print("eweights of each Edge formed by there neighbors: ", eweights)

partitions = []
partition_sets = []


# file1 = open('output_1.txt', "w")
# file2 = open('output_2.txt', "w")


def partition_graph(partition_count):
    n_cuts, membership = pymetis.part_graph(partition_count, adjacency=None, xadj=xadj, adjncy=adjncy, vweights=None,
                                            eweights=eweights, recursive=False)

    for i in range(partition_count):
        partition_data = np.argwhere(np.array(membership) == i).ravel()
        partitions.append(partition_data)
        partition_sets.append(set(partition_data))

        # print("Node of partition number {} : {}".format(i + 1, partitions[i]))
        # print("Node of partition number {} in set form : {}".format(i + 1, partition_sets[i]))

    print()


partition_graph(3)

for n, i in enumerate(partition_sets, 1):
    with open('new{}.txt'.format(n), 'w') as f:
        f.write(str(i))

# ---------------------------Read from text file ---------------------------------------
new_partition_set = []

# pathlib.Path().resolve()  (Read current path)
def read_text_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        new_partition_set.append(lines)


# iterate through all file
for file in os.listdir():
    # Check whether file is in text format or not
    if file.endswith(".txt"):
        file_path = f"{pathlib.Path().resolve()}/{file}"
        # call read text file function
        read_text_file(file_path)

# print("Partition Set", partition_sets)
# print("New Partition Set:-", new_partition_set)

new_partition_set = [item for sublist in new_partition_set for item in sublist]
# print("new partition set", new_partition_set)
# ---------------------------------------------------------------------------------------
""" ############### TASK6: Sampling Function #######################################  """
output = dict()
proxy_node_set = set()

# k = number of neighbors to sample
# G = FullGraph
# partition_sets = Store partition data
# sample_count = for every target node it will sample 15 sets of each containing 2 neighbors of target node
inter_avg = []
diff_avg = []
p = []
q = []
pp = []
flat_list = []

def sampling_function(G, alpha, k, partition_sets, xadj, adjncy, sample_count=15):
    for node in range(len(G.nodes)):
        neighbors_set = adjncy[xadj[node]:(xadj[node + 1])]
        # print()
        # print("Neighbors of node {} are".format(node), neighbors_set)

        node_in_partition_set = None

        for partition in partition_sets:
            if node in partition:
                node_in_partition_set = partition
                break

        inter = node_in_partition_set.intersection(neighbors_set)
        x = len(inter)
        diff = set(neighbors_set) - inter
        y = len(diff)

        # print("Own partition neighbors of node {} is:-".format(node), inter)
        inter_avg.append(len(inter))
        # print("Different partition neighbors of node {} is:-".format(node), diff)
        diff_avg.append(len(diff))

        w1 = []
        w2 = []
        for i in range(x):
            w1.append(alpha)
            # print("w1", alpha)

        for i in range(y):
            w2.append(1)
            # print("w2", 1)

        sample_picks = []
        for i in range(sample_count):
            # print("population", list(inter) + list(diff))
            # print("weights", list(w1) + list(w2))
            sample_pick = random.choices(population=list(inter) + list(diff), weights=list(w1) + list(w2), k=k)
            sample_picks.append(sample_pick)
            # sample_pick = random.choices(population=list(inter) + list(diff), k=k)
            # number of node which are in same partition
            own_partition_count = 0
            for i in sample_pick:
                if i in node_in_partition_set:
                    own_partition_count += 1

            # Proxy nodes are those node which are from different partition set
            # and less than own partition set
            if own_partition_count > 0 and own_partition_count < k:
                # print("Edge cut",sample_pick)
                proxy_node_set.add(frozenset(sample_pick))

            if node in output:
                output[node].append(sample_pick)
            else:
                output[node] = [sample_pick]

        flat_list = [item for sublist in sample_picks for item in sublist]

        # print("neighbors node after sampling", flat_list)

        def intersection(A, B):
            c_a = collections.Counter(A)
            c_b = collections.Counter(B)
            # print("Key value of node and number of times occur after sampling", c_a)
            # print("neighbors Nodes in same partition", c_b)
            # print("keys of total sampled list", c_a.keys())
            # print("keys of node in same partition", c_b.keys())

            a = c_a.keys()
            b = c_b.keys()
            c = a - b
            pp.append(c)

            # print("Proxy Node is", c)
            duplicates = []
            for c in c_a:
                duplicates += [c] * min(c_a[c], c_b[c])
                different = c_a.keys() - duplicates

        #     if len(different) == 0:
        #         print("q is empty", 0)
        #         print("p is", len(flat_list))
        #     else:
        #         q_count = []
        #         for i in set(diff):
        #             q_count.append(c_a.get(i))
        #         for g in flat_list:
        #             if g in c_a.keys() & c_b.keys():
        #                 (c_a.keys() & c_b.keys()).remove(g)
        #
        #         print("q is", sum(filter(None, list(q_count))))
        #         print("p is", len(flat_list) - sum(filter(None, list(q_count))))
        #     return duplicates
        #     print()
        #
        intersection(flat_list, inter)

        # own_partition = set(flat_list).intersection(inter)
        # own_partition = remove_duplicate(flat_list,inter)    #testing
        # print("own partition", own_partition)
        # p.append(len(own_partition))
        # print("p is :",p)
        # diff_partition = set(flat_list).intersection(diff)
        # #diff_partition = remove_duplicate(flat_list, diff)
        #
        # print("other partition", diff_partition)
        # q.append(len(diff_partition))
        # print("q is :", q)

    print()
    # print("list of inter nodes", list(inter))
    # print("list of other nodes", list(diff))
    print()

    print()
    # for node in output:
    #     print("Sample set for target node {} is {}".format(node, output[node]))
    # print()

    # for i in proxy_node_set:
    #     print(list(i))

    print()
    # print("total number of edge cut", len(proxy_node_set))
    flatten1 = [element for items in proxy_node_set for element in items]

    # print("Proxy node", set(flatten1))
    # print("number of proxy node", len(set(flatten1)))
    Proxy_node_flatten = [element for items in pp for element in items]
    # print("New Proxy node", set(Proxy_node_flatten))
    print("Length of new proxy nodes", len(set(Proxy_node_flatten)))


# sampling_function(G, 2, 2, partition_sets, xadj, adjncy, sample_count=15)


def updated_sampling_function(G, alpha, k, partition_sets, xadj, adjncy, sample_count=15):
    for node in range(len(G.nodes)):
        neighbors_set = adjncy[xadj[node]:(xadj[node + 1])]
        print()
        print("Neighbors of node {} are".format(node), neighbors_set)

        node_in_partition_set = None

        for partition in partition_sets:
            if node in partition:
                node_in_partition_set = partition
                break

        inter = set(node_in_partition_set).intersection(neighbors_set)
        x = len(inter)
        diff = set(neighbors_set) - inter
        y = len(diff)

        print("Own partition neighbors of node {} is:-".format(node), inter)
        inter_avg.append(len(inter))
        print("Different partition neighbors of node {} is:-".format(node), diff)
        diff_avg.append(len(diff))

        w1 = []
        w2 = []
        for i in range(x):
            w1.append(alpha)
            # print("w1", alpha)

        for i in range(y):
            w2.append(1)
            # print("w2", 1)

        sample_picks = []
        for i in range(sample_count):
            # print("population", list(inter) + list(diff))
            # print("weights", list(w1) + list(w2))
            sample_pick = random.choices(population=list(inter) + list(diff), weights=list(w1) + list(w2), k=k)
            sample_picks.append(sample_pick)
            # sample_pick = random.choices(population=list(inter) + list(diff), k=k)
            # number of node which are in same partition
            own_partition_count = 0
            for i in sample_pick:
                if i in node_in_partition_set:
                    own_partition_count += 1

            # Proxy nodes are those node which are from different partition set
            # and less than own partition set
            if own_partition_count > 0 and own_partition_count < k:
                # print("Edge cut",sample_pick)
                proxy_node_set.add(frozenset(sample_pick))

            if node in output:
                output[node].append(sample_pick)
            else:
                output[node] = [sample_pick]

        flat_list = [item for sublist in sample_picks for item in sublist]
        print("neighbors node after sampling", flat_list)

        def intersection(A, B):
            c_a = collections.Counter(A)
            c_b = collections.Counter(B)
            print("Key value of node and number of times occur after sampling", c_a)

            a = c_a.keys()
            b = c_b.keys()
            c = a - b
            pp.append(c)

            print("Proxy Node is", c)
            duplicates = []
            for c in c_a:
                duplicates += [c] * min(c_a[c], c_b[c])
                different = c_a.keys() - duplicates

            if len(different) == 0:
                print("q is empty", 0)
                print("p is", len(flat_list))
            else:
                q_count = []
                for i in set(diff):
                    q_count.append(c_a.get(i))
                for g in flat_list:
                    if g in c_a.keys() & c_b.keys():
                        (c_a.keys() & c_b.keys()).remove(g)

                print("q is", sum(filter(None, list(q_count))))
                print("p is", len(flat_list) - sum(filter(None, list(q_count))))
            return duplicates
            print()

        intersection(flat_list, inter)
    for node in output:
        print("Sample set for target node {} is {}".format(node, output[node]))
    print()

    for i in proxy_node_set:
        print(list(i))

    print()
    print("total number of edge cut", len(proxy_node_set))
    flatten1 = [element for items in proxy_node_set for element in items]

    print("Proxy node", set(flatten1))
    print("number of proxy node", len(set(flatten1)))
    Proxy_node_flatten = [element for items in pp for element in items]
    print("New Proxy node after optimization", set(Proxy_node_flatten))
    print("Length of new proxy nodes after optimization", len(set(Proxy_node_flatten)))


def optimized_partition(partition_sets, sample_length, xadj, adjncy, alpha, k):
    traversed_node = set()
    print("partition sets", partition_sets)
    for i in range(len(partition_sets)):
        print("i is:", i)

        for node in partition_sets[i]:
            if node not in traversed_node:
                traversed_node.add(node)
                print("Current node is", node)
                print("same partition is", partition_sets[i])
                node_in_partition_set = None

                for partition in partition_sets:
                    if node in partition:
                        node_in_partition_set = partition
                        break
                neighbors_set = adjncy[xadj[node]:(xadj[node + 1])]
                print("Neighbors of node {} are".format(node), neighbors_set)

                inter = set(node_in_partition_set).intersection(neighbors_set)
                x = len(inter)
                diff = set(neighbors_set) - inter
                y = len(diff)

                print("Own partition neighbors of node {} is:-".format(node), inter)
                print("Different partition neighbors of node {} is:-".format(node), diff)

                w1 = []
                w2 = []
                for a in range(x):
                    w1.append(alpha)
                    # print("w1", alpha)

                for b in range(y):
                    w2.append(1)

                samples = []
                for c in range(sample_length):
                    sample_pick = random.choices(population=list(inter) + list(diff), weights=list(w1) + list(w2), k=k)
                    samples.append(sample_pick)

                print("sample picks", samples)

                flatlist = [element for sublist in samples for element in sublist]
                print("Flatlist", flatlist)

                def intersection(A, B):
                    c_a = collections.Counter(A)
                    c_b = collections.Counter(B)
                    print("Key value of node and number of times occur after sampling", c_a)

                    a = c_a.keys()
                    b = c_b.keys()
                    c = a - b
                    pp.append(c)

                    print("Proxy Node is", c)
                    duplicates = []
                    for c in c_a:
                        duplicates += [c] * min(c_a[c], c_b[c])
                        different = c_a.keys() - duplicates

                    if len(different) == 0:
                        print("q is empty", 0)
                        print("p is", len(flatlist))
                    else:
                        q_count = []
                        for i in set(diff):
                            q_count.append(c_a.get(i))
                        for g in flatlist:
                            if g in c_a.keys() & c_b.keys():
                                (c_a.keys() & c_b.keys()).remove(g)

                        print("q is", sum(filter(None, list(q_count))))
                        print("p is", len(flatlist) - sum(filter(None, list(q_count))))
                    return duplicates
                    print()

                intersection(flatlist, inter)

                samples = np.unique(flatlist)
                print("Unique and Flatten samples", samples)

                for sample in samples:
                    print("current sample", sample)

                    if sample not in partition_sets[i]:
                        diff_partition_set_index = find_partition_index(partition_sets, sample)
                        print("Index of different partition node ", diff_partition_set_index)
                        print("next", partition_sets[diff_partition_set_index])
                        print("sample set", [sample])
                        same_partition = partition_sets[i]
                        same_partition.append(sample)
                        diff_partition = list(set(partition_sets[diff_partition_set_index]) - set([sample]))
                        partition_sets[diff_partition_set_index] = diff_partition
                        partition_sets[i] = same_partition
                        print("new different partition ", partition_sets[diff_partition_set_index])
                        print("new same partition", partition_sets[i])

                print("=====================================================================")

    return partition_sets


def find_partition_index(partition_sets, node):
    for i in range(len(partition_sets)):
        if node in partition_sets[i]:
            return i
    return None


""" Statistics """


# print("inner average of neighbors node before sampling", inter_avg)
# print("outer average of neighbors node before sampling", diff_avg)
# sum_of_diff = sum(diff_avg)
# total_before_sampling = sum(inter_avg) + sum_of_diff
# print("Average number of nodes of neighbors in same partitions", round(statistics.mean(inter_avg), 2))
# print("Average number of nodes of neighbors in different partitions", round(statistics.mean(diff_avg), 2))
# print("percentage of node have neighbors going to the other partition : ",
#       round(sum_of_diff / total_before_sampling * 100, 2))
#
# sum_q = sum(q)
# total = sum(p) + sum_q
#
# print()
# print("inner average of neighbors node after sampling", p)
# print("outer average of neighbors node after sampling", q)
# print()
# print("percentage of node have neighbors going to the other partition : ", round(sum_q/total*100, 2))
# print("Average number of Node of neighbors in same partition after Sampling", round(statistics.mean(p),2))
# print("Average number of Node of neighbors in other partition after Sampling", round(statistics.mean(q),2))

# partition_list = []
# for partition in partition_sets:
#     partition_list.append(list(partition))
#
# new_partition_set = optimized_partition(partition_sets=partition_list, sample_length=15, xadj=xadj, adjncy=adjncy,
#                                         alpha=2, k=2)
#
# print("Final optimized partition set",new_partition_set)

def optimized_sampling_function(G, alpha, k, partition_sets, xadj, adjncy, sample_count=15):
    D_list = []

    for p in range(len(partition_sets)):
        D_list.append(set())

    print("Dlist is",D_list)
    for node in range(len(G.nodes)):
        neighbors_set = adjncy[xadj[node]:(xadj[node + 1])]
        # print()
        # print("Neighbors of node {} are".format(node), neighbors_set)

        node_in_partition_set = None
        same_partition_set_index = None

        for partition_index in range(len(partition_sets)):
            if node in partition_sets[partition_index]:
                node_in_partition_set = partition_sets[partition_index]
                same_partition_set_index = partition_index
                break

        same_partition_dynamic_set = D_list[same_partition_set_index]
        same_partition_dynamic_size = len(same_partition_dynamic_set)

        inter = node_in_partition_set.intersection(neighbors_set)
        x = len(inter)

        diff = set(neighbors_set) - inter
        y = len(diff)
        # print("-------------------------------")
        # print("Own partition neighbors of node {} is:-".format(node), inter)
        # inter_avg.append(len(inter))
        # print("Different partition neighbors of node {} is:-".format(node), diff)
        # diff_avg.append(len(diff))
        # print("X is ", x)
        # print("Y is", y)
        # print("Size of Dynamic list", same_partition_dynamic_size)
        # print("D list", D_list)
        # print("proxy length of optimized", len(D_list[0]) + len(D_list[1]))
        w1 = []
        w2 = []

        for i in range(x):
            w1.append(alpha / 10 / (x + same_partition_dynamic_size))

        for i in range(y):
            w2.append((alpha / 10 / (x + same_partition_dynamic_size)) / (len(neighbors_set)))
            # w2.append((1 - alpha/10)/(len(neighbors_set) + y))
            # w2.append((1 - alpha/10)/y)

        # print("weights for node {} is {}".format(node, list(w1) + list(w2)))
        sample_picks = []
        for i in range(sample_count):
            # print("population", list(inter) + list(diff))
            # print("weights", list(w1) + list(w2))
            sample_pick = random.choices(population=list(inter) + list(diff), weights=list(w1) + list(w2), k=k)
            sample_picks.append(sample_pick)

            for sampled_node in sample_pick:
                if sampled_node not in node_in_partition_set:
                    D_list[same_partition_set_index].add(sampled_node)
                # else:
                #     for partition_index in range(len(partition_sets)):
                #         if sampled_node in partition_sets[partition_index]:
                #             diff_partition_index = partition_index
                #             break
                #     # find partition set in which sample node belongs to
                #     # find partition index of this partition = diff_partition_index
                #     D_list[diff_partition_index].add(sampled_node)
            # sample_pick = random.choices(population=list(inter) + list(diff), k=k)
            # number of node which are in same partition
            own_partition_count = 0
            for i in sample_pick:
                if i in node_in_partition_set:
                    own_partition_count += 1

            if node in output:
                output[node].append(sample_pick)
            else:
                output[node] = [sample_pick]

        flat_list = [item for sublist in sample_picks for item in sublist]

        # print("neighbors", neighbors_set)
        # print("sample picked", sample_picks)
        # print("neighbors node after sampling", flat_list)

        def intersection(A, B):
            c_a = collections.Counter(A)
            c_b = collections.Counter(B)
            # print("Key value of node and number of times occur after sampling", c_a)
            # print("neighbors Nodes in same partition", c_b)
            # print("keys of total sampled list", c_a.keys())
            # print("keys of node in same partition", c_b.keys())

            a = c_a.keys()
            b = c_b.keys()
            c = a - b
            pp.append(c)

            # print("Proxy Node is", c)
            duplicates = []
            for c in c_a:
                duplicates += [c] * min(c_a[c], c_b[c])
                different = c_a.keys() - duplicates

            # if len(different) == 0:
            #     print("q is empty", 0)
            #     print("p is", len(flat_list))
            # else:
            #     q_count = []
            #     for i in set(diff):
            #         q_count.append(c_a.get(i))
            #     for g in flat_list:
            #         if g in c_a.keys() & c_b.keys():
            #             (c_a.keys() & c_b.keys()).remove(g)
            #
            #     print("q is", sum(filter(None, list(q_count))))
            #     print("p is", len(flat_list) - sum(filter(None, list(q_count))))
            # return duplicates
            # print()

        intersection(flat_list, inter)

        # own_partition = set(flat_list).intersection(inter)
        # own_partition = remove_duplicate(flat_list,inter)    #testing
        # print("own partition", own_partition)
        # p.append(len(own_partition))
        # print("p is :",p)
        # diff_partition = set(flat_list).intersection(diff)
        # #diff_partition = remove_duplicate(flat_list, diff)
        #
        # print("other partition", diff_partition)
        # q.append(len(diff_partition))
        # print("q is :", q)

    print()
    # print("list of inter nodes", list(inter))
    # print("list of other nodes", list(diff))
    print()

    print()
    # for node in output:
    #     print("Sample set for target node {} is {}".format(node, output[node]))
    # print()

    # for i in proxy_node_set:
    #     print(list(i))

    # print()
    # print("total number of edge cut", len(proxy_node_set))
    flatten1 = [element for items in proxy_node_set for element in items]

    # print("Proxy node", set(flatten1))
    # print("number of proxy node", len(set(flatten1)))
    Proxy_node_flatten = [element for items in pp for element in items]
    # print("New Proxy node", set(Proxy_node_flatten))
    # print("Length of optimized proxy nodes", len(D_list))
    # print("Length of optimized proxy nodes", len(set(Proxy_node_flatten)))
    print("proxy length of optimized", len(D_list[0]) + len(D_list[1]))


sampling_function(G, 10, 15, partition_sets, xadj, adjncy, sample_count=15)
optimized_sampling_function(G, 10, 15, partition_sets, xadj, adjncy, sample_count=15)

# sampling_function(G, 10, 15, new_partition_set, xadj, adjncy, sample_count=15)
# optimized_sampling_function(G, 10, 15, new_partition_set, xadj, adjncy, sample_count=15)


##Model

class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """

    def __init__(self, features, cuda=False, gcn=False):
        """
        Initializes the aggregator for a specific graph.
        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()

        self.features = features
        self.cuda = cuda
        self.gcn = gcn

    def forward(self, nodes, to_neighs, num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # # Local pointers to functions (speed hack)
        # _set = set
        # if not num_sample is None:
        #     _sample = random.sample
        #     samp_neighs = [_set(_sample(to_neigh,
        #                                 num_sample,
        #                                 )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        # else:
        #     samp_neighs = to_neighs
        for node in range(len(G.nodes)):
            neighbors_set = adjncy[xadj[node]:(xadj[node + 1])]
            # print()
            # print("Neighbors of node {} are".format(node), neighbors_set)

            node_in_partition_set = None

            for partition in partition_sets:
                if node in partition:
                    node_in_partition_set = partition
                    break

            inter = node_in_partition_set.intersection(neighbors_set)
            x = len(inter)
            diff = set(neighbors_set) - inter
            y = len(diff)

            # print("Own partition neighbors of node {} is:-".format(node), inter)
            inter_avg.append(len(inter))
            # print("Different partition neighbors of node {} is:-".format(node), diff)
            diff_avg.append(len(diff))

            w1 = []
            w2 = []
            for i in range(x):
                w1.append(5)
                # print("w1", alpha)

            for i in range(y):
                w2.append(1)
                # print("w2", 1)

            sample_picks = []
            for i in range(15):
                # print("population", list(inter) + list(diff))
                # print("weights", list(w1) + list(w2))
                sample_pick = random.choices(population=list(inter) + list(diff), weights=list(w1) + list(w2), k=5)
                sample_picks.append(sample_pick)
                # sample_pick = random.choices(population=list(inter) + list(diff), k=k)
                # number of node which are in same partition
                own_partition_count = 0
                for i in sample_pick:
                    if i in node_in_partition_set:
                        own_partition_count += 1

            flat_list = [item for sublist in sample_picks for item in sublist]
        return flat_list
        # print("sample", flat_list)
        # if self.gcn:
        #     flat_list = [flat_list + set([nodes[i]]) for i, samp_neigh in enumerate(flat_list)]
        # unique_nodes_list = list(set.union(*flat_list))
        # print("unique node list", unique_nodes_list)
        # print()
        #  print ("\n unl's size=",len(unique_nodes_list))
        # unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        # mask = Variable(torch.zeros(len(flat_list), len(unique_nodes)))
        # column_indices = [unique_nodes[n] for samp_neigh in flat_list for n in samp_neigh]
        # row_indices = [i for i in range(len(flat_list)) for j in range(len(flat_list[i]))]
        # mask[row_indices, column_indices] = 1
        # if self.cuda:
        #     mask = mask.cuda()
        # num_neigh = mask.sum(1, keepdim=True)
        # mask = mask.div(num_neigh)
        # if self.cuda:
        #     embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        # else:
        #     embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
        # to_feats = mask.mm(embed_matrix)
        # return to_feats

class Encoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """

    def __init__(self, features, feature_dim,
                 embed_dim, adj_lists, aggregator,
                 num_sample=10,
                 base_model=None, gcn=False, cuda=False,
                 feature_transform=False):
        super(Encoder, self).__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample
        if base_model != None:
            self.base_model = base_model

        self.gcn = gcn
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.aggregator.cuda = cuda
        self.weight = nn.Parameter(
            torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes.
        nodes     -- list of nodes
        """
        neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes],
                                              self.num_sample)

        print("neigh_feats", neigh_feats)
        if not self.gcn:
            if self.cuda:
                self_feats = self.features(torch.LongTensor(nodes).cuda())
            else:
                self_feats = self.features(torch.LongTensor(nodes))
            combined = torch.cat([self_feats, neigh_feats], dim=1)
        else:
            combined = neigh_feats

        print("Combined", combined)
        # combined = F.relu(self.weight.mm(combined.t()))
        return combined


class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())


def run_gitml():
    np.random.seed(1)
    random.seed(1)
    num_nodes = 37700
    num_feats = 4004
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    feat_data = np.zeros((num_nodes, num_feats))
    features = nn.Embedding(2708, 1433)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    print("Feature weight is", features.weight)
    agg1 = MeanAggregator(features, cuda=False)
    enc1 = Encoder(features, 1433, 128, adjacency_list, agg1, gcn=True, cuda=False)
    print("encoder", enc1)
    graphsage = SupervisedGraphSage(7, enc1)
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    for batch in range(10):
        batch_nodes = train[:256]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes,
                              Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time - start_time)
        print(batch, loss.item())

    val_output = graphsage.forward(val)
    print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    print("Average batch time:", np.mean(times))


# if __name__ == "__main__":
#     run_gitml()