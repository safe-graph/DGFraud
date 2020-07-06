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

def load_data_dblp(prefix='./example_data/', file_name = 'YelpChi.mat'):
    training_size = 0.8
    data = sio.loadmat(prefix + file_name)
    truelabels, features = data['label'], data['features'].astype(float)
    truelabels = truelabels.tolist()[0]
    features = features.todense()
    N = features.shape[0]
    rownetworks = [data['net_rur']]
    # rownetworks = [data['net_APA'] - np.eye(N), data['net_APCPA'] - np.eye(N), data['net_APTPA'] - np.eye(N)]
    index = range(len(truelabels))
    train_num = int(len(truelabels) * training_size)
    train_idx = set(np.random.choice(index, train_num, replace=False))
    test_idx = set(index).difference(train_idx)
    # X_train, X_test, y_train, y_test = train_test_split(index, y, stratify=y, test_size=0.4, random_state=48,
    #                                                     shuffle=True)
    return rownetworks, features, truelabels, train_idx, test_idx

def load_data(prefix='./example_data/', file_name = 'YelpChi.mat', normalize=True, load_walks=False):
    adjs, feats, truelabels, train_idx, test_idx = load_data_dblp(prefix, file_name)
    gs = [nx.to_networkx_graph(adj) for adj in adjs]
    id_map = {int(i):i for i in range(len(truelabels))}
    class_map = {int(i):truelabels[i] for i in range(len(truelabels))}
    walks = []
    G = gs[0] # change the index to specify which adj matrix to use for aggregation
    for node in G.nodes():
        G.node[node]['feature'] = feats[node,:].tolist()[0]
        G.node[node]['label'] = [truelabels[node]]
        if node in train_idx:
            G.node[node]['test'] = False
            G.node[node]['val'] = False
        else:
            G.node[node]['test'] = True
            G.node[node]['val'] = True
    broken_count = 0
    for node in G.nodes():
        if not 'val' in G.node[node] or not 'test' in G.node[node]:
            G.remove_node(node)
            broken_count += 1
    print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))
    print("Loaded data.. now preprocessing..")
    for edge in G.edges():
        G[edge[0]][edge[1]]['train_removed'] = False

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

    return G, feats, id_map, walks, class_map



def load_data_ori(prefix, normalize=True, load_walks=False):
    G_data = json.load(open(prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    if isinstance(G.nodes()[0], int):
        conversion = lambda n : int(n)
    else:
        conversion = lambda n : n

    if os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy")
    else:
        print("No features present.. Only identity features will be used.")
        feats = None
    id_map = json.load(open(prefix + "-id_map.json"))
    id_map = {conversion(k):int(v) for k,v in id_map.items()}
    walks = []
    class_map = json.load(open(prefix + "-class_map.json"))
    if isinstance(list(class_map.values())[0], list):
        lab_conversion = lambda n : n
    else:
        lab_conversion = lambda n : int(n)

    class_map = {conversion(k):lab_conversion(v) for k,v in class_map.items()}

    ## Remove all nodes that do not have val/test annotations
    ## (necessary because of networkx weirdness with the Reddit data)
    broken_count = 0
    for node in G.nodes():
        if not 'val' in G.node[node] or not 'test' in G.node[node]:
            G.remove_node(node)
            broken_count += 1
    print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))

    ## Make sure the graph has edge train_removed annotations
    ## (some datasets might already have this..)
    print("Loaded data.. now preprocessing..")
    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
            G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        train_ids = np.array([id_map[n] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']])
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)
    
    if load_walks:
        with open(prefix + "-walks.txt") as fp:
            for line in fp:
                walks.append(map(conversion, line.split()))

    return G, feats, id_map, walks, class_map

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
