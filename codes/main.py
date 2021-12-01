# libraries
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

import graph_partition
import preprocessing_step
import sampling_method
import optimized_sampling_method


graph_partition.partition_graph(2)
G = partition_graph.G
xadj = graph_partition.xadj
partition_sets = graph_partition.partition_sets
adjncy = graph_partition.adjncy
new_partition_sets = optimized_sampling_method.new_partition_sets

sampling_method.sampling_function(G, alpha=2, k=2, partition_sets=partition_sets, xadj=xadj, adjncy=adjncy, sample_count=15)
optimized_sampling_method.optimized_partition(partition_sets=new_partition_sets, sample_count=15, xadj=xadj, adjncy=adjncy,
                                        alpha=2, k=2)
