from __future__ import print_function

import numpy as np
import random
import json
import sys
import os
import scipy.io as sio

import networkx as nx
from networkx.readwrite import json_graph
version_info = list(map(int, nx.__version__.split('.')))
major = version_info[0]
minor = version_info[1]
assert (major <= 1) and (minor <= 11), "networkx major version > 1.11"


WALK_LEN=5  
N_WALKS=50

def load_mat_full(prefix='./example_data/', file_name = 'YelpChi.mat', relations=['net_rur'], train_size=0.8):
    data = sio.loadmat(prefix + file_name)
    truelabels, features = data['label'], data['features'].astype(float)
    truelabels = truelabels.tolist()[0]
    features = features.todense()
    N = features.shape[0]
    adj_mat = [data[relation] for relation in relations]
    index = range(len(truelabels))
    train_num = int(len(truelabels) * 0.8)
    train_idx = set(np.random.choice(index, train_num, replace=False))
    test_idx = set(index).difference(train_idx)
    train_num = int(len(truelabels) * train_size)
    train_idx = set(list(train_idx)[:train_num])
    return adj_mat, features, truelabels, train_idx, test_idx

def graph_process(graph, features, truelabels, test_idx):
    print('-------processing graph-------------')
    for node in graph.nodes():
        graph.node[node]['feature'] = features[node,:].tolist()[0]
        graph.node[node]['label'] = [truelabels[node]]
        if node in test_idx:
            graph.node[node]['test'] = True
            graph.node[node]['val'] = True
        else:
            graph.node[node]['test'] = False
            graph.node[node]['val'] = False
    broken_count = 0
    for edge in graph.edges():
        graph[edge[0]][edge[1]]['train_removed'] = False
    return graph

def load_data(prefix='./example_data/', file_name = 'YelpChi.mat', relations=['net_rur'], normalize=True, load_walks=False, train_size=0.8):
    adjs, feats, truelabels, train_idx, test_idx = load_mat_full(prefix, file_name, relations, train_size)
    gs = [nx.to_networkx_graph(adj) for adj in adjs]
    id_map = {int(i):i for i in range(len(truelabels))}
    class_map = {int(i):truelabels[i] for i in range(len(truelabels))}
    walks = []
    adj_main = np.sum(adjs) # change the index to specify which adj matrix to use for aggregation
    G = nx.to_networkx_graph(adj_main)
    gs = [graph_process(g, feats, truelabels, test_idx) for g in gs]
    G = graph_process(G, feats, truelabels, test_idx)
    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        train_ids = np.array([id_map[n] for n in G.nodes()])
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)
    if load_walks:
        with open(prefix + "-walks.txt") as fp:
            for line in fp:
                walks.append(map(conversion, line.split()))
    return G, feats, id_map, walks, class_map, gs

def run_random_walks(G, nodes, num_walks=N_WALKS):
    pairs = []
    for count, node in enumerate(nodes):
        if G.degree(node) == 0:
            continue
        for i in range(num_walks):
            curr_node = node
            for j in range(WALK_LEN):
                next_node = random.choice(G.neighbors(curr_node))
                # self co-occurrences are useless
                if curr_node != node:
                    pairs.append((node,curr_node))
                curr_node = next_node
        if count % 1000 == 0:
            print("Done walks for", count, "nodes")
    return pairs

if __name__ == "__main__":
    """ Run random walks """
    graph_file = sys.argv[1]
    out_file = sys.argv[2]
    G_data = json.load(open(graph_file))
    G = json_graph.node_link_graph(G_data)
    nodes = [n for n in G.nodes() if not G.node[n]["val"] and not G.node[n]["test"]]
    G = G.subgraph(nodes)
    pairs = run_random_walks(G, nodes)
    with open(out_file, "w") as fp:
        fp.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in pairs]))
