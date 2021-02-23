import dgl
import networkx as nx
import torch as th
from dgl.data.utils import load_graphs
import os

a = load_graphs("../data_bin/dgl/" + "all" + ".bin")
print(a)