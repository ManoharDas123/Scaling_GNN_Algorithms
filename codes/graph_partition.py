import pymetis
import networkx as nx
import numpy as np
from tqdm import tqdm
import pandas as pd
import json

# Declare Variables
partitions = []
partition_sets = []
eweights = []
adj = {}
adjncy = []
xadj = [0]
size = 1
K = 5
no_of_partition = 2


def findDegree(graph):
    return [val for (n, val) in graph.degree()]


def weight(node, sampling_length, degrees):
    probability_weight = 1 - pow((1 - 1 / degrees[node]), sampling_length)
    return int(probability_weight * 100)


degrees = findDegree(G)
print('Degrees of Each Node is:-', degrees)
adjacency_list = dict(G.adjacency())

for node in adjacency_list:
    adjacency = list(adjacency_list[node].keys())
    adj[node] = adjacency
    adjncy += adjacency
    for neighbor in adjacency:
        eweights.append(weight(neighbor, K, degrees))
    xadj.append(xadj[size - 1] + len(adjacency))
    size += 1


def partition_graph(partition_count):
    n_cuts, membership = pymetis.part_graph(partition_count, adjacency=None, xadj=xadj, adjncy=adjncy, vweights=None,
                                            eweights=eweights, recursive=False)

    for i in range(partition_count):
        partition_data = np.argwhere(np.array(membership) == i).ravel()
        partitions.append(partition_data)
        partition_sets.append(set(partition_data))
        print("Node of partition number {} : {}".format(i + 1, partitions[i]))
        print("Node of partition number {} in set form : {}".format(i + 1, partition_sets[i]))

    print()


partition_graph(no_of_partition)

for i in partition_sets:
    print("partition sets", list(i))
