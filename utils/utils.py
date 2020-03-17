import random
import scipy.io as sio
import scipy.sparse as sp
import numpy as np


# symmetrically normalize adjacency matrix
def normalize_adj(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).A


# Construct feed dictionary
def construct_feed_dict(x, a, t, b, learning_rate, momentum, placeholders):
    feed_dict = dict()
    feed_dict.update({placeholders['x']: x})
    feed_dict.update({placeholders['a']: a})
    feed_dict.update({placeholders['t']: t})
    feed_dict.update({placeholders['batch_index']: b})
    feed_dict.update({placeholders['lr']: learning_rate})
    feed_dict.update({placeholders['mom']: momentum})
    feed_dict.update({placeholders['num_features_nonzero']: x[1].shape})
    return feed_dict


# Construct feed dictionary for SemiGNN
def construct_feed_dict_sg(x, a, u_i, u_j, batch_graph_label, batch_data, batch_sup_label, learning_rate, momentum,
                           placeholders):
    feed_dict = dict()
    feed_dict.update({placeholders['x']: x})
    feed_dict.update({placeholders['a']: a})
    feed_dict.update({placeholders['u_i']: u_i})
    feed_dict.update({placeholders['u_j']: u_j})
    feed_dict.update({placeholders['graph_t']: batch_graph_label})
    feed_dict.update({placeholders['batch_index']: batch_data})
    feed_dict.update({placeholders['sup_t']: batch_sup_label})
    feed_dict.update({placeholders['lr']: learning_rate})
    feed_dict.update({placeholders['mom']: momentum})
    feed_dict.update({placeholders['num_features_nonzero']: x[1].shape})
    return feed_dict


def pad_adjlist(x_data):
    # Get lengths of each row of data
    lens = np.array([len(x_data[i]) for i in range(len(x_data))])

    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:, None]

    # Setup output array and put elements from data into masked positions
    padded = np.zeros(mask.shape)
    for i in range(mask.shape[0]):
        padded[i] = np.random.choice(x_data[i], mask.shape[1])
    padded[mask] = np.hstack((x_data[:]))
    return padded


def matrix_to_adjlist(M, pad=True):
    adjlist = []
    for i in range(len(M)):
        adjline = [i]
        for j in range(len(M[i])):
            if M[i][j] == 1:
                adjline.append(j)
        adjlist.append(adjline)
    if pad:
        adjlist = pad_adjlist(adjlist)
    return adjlist


def adjlist_to_matrix(adjlist):
    nodes = len(adjlist)
    M = np.zeros((nodes, nodes))
    for i in range(nodes):
        for j in adjlist[i]:
            M[i][j] = 1
    return M


def pairs_to_matrix(pairs, nodes):
    M = np.zeros((nodes, nodes))
    for i, j in pairs:
        M[i][j] = 1
    return M


# Random walk on graph
def generate_random_walk(adjlist, start, walklength):
    t = 1
    walk_path = np.array([start])
    while t <= walklength:
        neighbors = adjlist[start]
        current = np.random.choice(neighbors)
        walk_path = np.append(walk_path, current)
        start = current
        t += 1
    return walk_path


def random_walk_sampling(adjlist, numerate, walklength):
    nodes = range(0, len(adjlist))  # node index starts from zero
    walks = []
    for n in range(numerate):
        for node in nodes:
            walks.append(generate_random_walk(adjlist, node, walklength))
    pairs = []
    for i in range(len(walks)):
        for j in range(1, len(walks[i])):
            pair = [walks[i][0], walks[i][j]]
        pairs.append(pair)
    return pairs


def negative_sampling(adj_nodelist):
    degree = [len(neighbors) for neighbors in adj_nodelist]
    node_negative_distribution = np.power(np.array(degree, dtype=np.float32), 0.75)
    node_negative_distribution /= np.sum(node_negative_distribution)
    node_sampling = AliasSampling(prob=node_negative_distribution)
    return node_negative_distribution, node_sampling


def get_batch_negative_sampling(ix, int_batch, pairs, adj_nodelist, sup_label, Q=3, node_sampling='atlas'):
    pairs_size = len(pairs)
    if ix + int_batch >= pairs_size:
        ix = pairs_size - int_batch
        end = pairs_size
    else:
        end = ix + int_batch
    u_i = []
    u_j = []
    graph_label = []
    num_of_nodes = len(adj_nodelist)
    node_negative_distribution, nodesampling = negative_sampling(adj_nodelist)
    for index in range(ix, end):
        u_i.append(pairs[index][0])
        u_j.append(pairs[index][1])
        graph_label.append(1)
        for i in range(Q):
            while True:
                if node_sampling == 'numpy':
                    negative_node = np.random.choice(num_of_nodes, node_negative_distribution)
                    if negative_node not in adj_nodelist[pairs[index][0]]:
                        break
                elif node_sampling == 'atlas':
                    negative_node = nodesampling.sampling()
                    if negative_node not in adj_nodelist[pairs[index][0]]:
                        break
                elif node_sampling == 'uniform':
                    negative_node = np.random.randint(0, num_of_nodes)
                    if negative_node not in adj_nodelist[pairs[index][0]]:
                        break
            u_i.append(pairs[index][0])
            u_j.append(negative_node)
            graph_label.append(-1)
    graph_label = np.array(graph_label)
    graph_label = graph_label.reshape(graph_label.shape[0], 1)
    train_index = np.array(list(set(u_i)))
    train_sup_label = np.array([sup_label[ti] for ti in train_index])
    return u_i, u_j, graph_label, train_index, train_sup_label


# Reference: https://en.wikipedia.org/wiki/Alias_method
class AliasSampling:

    def __init__(self, prob):
        self.n = len(prob)
        self.U = np.array(prob) * self.n
        self.K = [i for i in range(len(prob))]
        overfull, underfull = [], []
        for i, U_i in enumerate(self.U):
            if U_i > 1:
                overfull.append(i)
            elif U_i < 1:
                underfull.append(i)
        while len(overfull) and len(underfull):
            i, j = overfull.pop(), underfull.pop()
            self.K[j] = i
            self.U[i] = self.U[i] - (1 - self.U[j])
            if self.U[i] > 1:
                overfull.append(i)
            elif self.U[i] < 1:
                underfull.append(i)

    def sampling(self, n=1):
        x = np.random.rand(n)
        i = np.floor(self.n * x)
        y = self.n * x - i
        i = i.astype(np.int32)
        res = [i[k] if y[k] < self.U[i[k]] else self.K[i[k]] for k in range(n)]
        if n == 1:
            return res[0]
        else:
            return res
