import dgl
import networkx as nx
import torch as th
from dgl.data.utils import save_graphs
import os

'''
names1 = ["dblp_netrep", "ca-GrQc", "ca-HepPh"]
names2 = ["livejournal", "dblp_netrep", "dblp_snap", "hindex_top", "ca-GrQc", "ca-HepPh", "euall"]
g_list = []
graph_size = []
for name in names1:
    nxg = nx.read_edgelist(name + ".edges")
    print(name)
    g = dgl.DGLGraph()
    g.from_networkx(nxg)
    g_list.append(g)
    graph_size.append(len(nxg.nodes()))
graph_labels = {"graph_sizes": th.tensor(graph_size)}
save_graphs("../data_bin/imdb_multi.bin", g_list, graph_labels)

g_list = []
graph_size = []
for name in names2:
    nxg = nx.read_edgelist(name + ".edges")
    print(name)
    g = dgl.DGLGraph()
    g.from_networkx(nxg)
    g_list.append(g)
    graph_size.append(len(nxg.nodes()))
graph_labels = {"graph_sizes": th.tensor(graph_size)}
save_graphs("../data_bin/imdb_binary.bin", g_list, graph_labels)
'''

#names1 = ["livejournal", "facebook", "imdb", "dblp_netrep", "dblp_snap", "academia", "euall", "google", "flickr", "pokec", "amazon", "twitter", "youtube"]
names1 = ['livejournal', 'facebook', 'imdb', 'dblp_netrep', 'dblp_snap', 'academia', 'youtube', 'twitter', 'amazon', 'pokec', 'google', 'euall', 'enron', 'flickr', 'git', 'hindex_top', 'ca-AstroPh', 'ca-CondMat', 'ca-GrQc', 'ca-HepPh', 'ca-HepTh', 'cora', 'citeseer', 'wiki', 'terrorist', 'usa', 'brazi', 'europe']
#names1 = ["imdb", "dblp_netrep", "dblp_snap", "academia", "ca-HepPh", "euall", "google", "flickr"]
names1 = ["cora", "citeseer"]
g_list = []
graph_size = []
for name in names1:    
    nxg = nx.read_edgelist("../../../data/processed/" + name + ".edges")
    print(name)
    g = dgl.DGLGraph()
    g.from_networkx(nxg)
    g_list.append(g)
    graph_size.append(len(nxg.nodes()))
graph_labels = {"graph_sizes": th.tensor(graph_size)}
save_graphs("../data_bin/dgl/" + "test" + ".bin", g_list, graph_labels)

