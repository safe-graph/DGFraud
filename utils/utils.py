import random
import scipy.io as sio
import scipy.sparse as sp
import numpy as np


def sparse_to_tuple(sparse_mx):
	"""
	Convert sparse matrix to tuple representation.
	"""
	def to_tuple(mx):
		if not sp.isspmatrix_coo(mx):
			mx = mx.tocoo()
		coords = np.vstack((mx.row, mx.col)).transpose()
		values = mx.data
		shape = mx.shape
		return coords, values, shape

	if isinstance(sparse_mx, list):
		for i in range(len(sparse_mx)):
			sparse_mx[i] = to_tuple(sparse_mx[i])
	else:
		sparse_mx = to_tuple(sparse_mx)

	return sparse_mx


def normalize_adj(adj):
	"""
	Symmetrically normalize adjacency matrix
	"""
	adj = sp.coo_matrix(adj)
	rowsum = np.array(adj.sum(1))
	d_inv_sqrt = np.power(rowsum, -0.5).flatten()
	d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
	d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
	return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
	"""
	Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation.
	"""
	adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
	return sparse_to_tuple(adj_normalized)


def preprocess_feature(features, to_tuple=True):
	"""
	Row-normalize feature matrix and convert to tuple representation
	"""

	features = sp.lil_matrix(features)
	rowsum = np.array(features.sum(1))
	r_inv = np.power(rowsum, -1).flatten()
	r_inv[np.isinf(r_inv)] = 0.
	r_mat_inv = sp.diags(r_inv)
	features = r_mat_inv.dot(features)

	if to_tuple:
		return sparse_to_tuple(features)
	else:
		return features


# Construct feed dictionary for SemiGNN
def construct_feed_dict_semi(a, u_i, u_j, batch_graph_label, batch_data, batch_sup_label, learning_rate, momentum,
						   placeholders):
	feed_dict = dict()
	feed_dict.update({placeholders['a']: a})
	feed_dict.update({placeholders['u_i']: u_i})
	feed_dict.update({placeholders['u_j']: u_j})
	feed_dict.update({placeholders['graph_label']: batch_graph_label})
	feed_dict.update({placeholders['batch_index']: batch_data})
	feed_dict.update({placeholders['sup_label']: batch_sup_label})
	feed_dict.update({placeholders['lr']: learning_rate})
	feed_dict.update({placeholders['mom']: momentum})
	return feed_dict


def sample_mask(idx, l):
	"""
	Create mask.
	"""
	mask = np.zeros(l)
	mask[idx] = 1
	return np.array(mask, dtype=np.bool)


# Construct feed dictionary for SemiGNN
def construct_feed_dict_spam(h, adj_info, t, b,  learning_rate, momentum, placeholders):
	feed_dict = dict()
	feed_dict.update({placeholders['user_review_adj']: adj_info[0]})
	feed_dict.update({placeholders['user_item_adj']: adj_info[1]})
	feed_dict.update({placeholders['item_review_adj']: adj_info[2]})
	feed_dict.update({placeholders['item_user_adj']: adj_info[3]})
	feed_dict.update({placeholders['review_user_adj']: adj_info[4]})
	feed_dict.update({placeholders['review_item_adj']: adj_info[5]})
	feed_dict.update({placeholders['homo_adj']: adj_info[6]})
	feed_dict.update({placeholders['review_vecs']: h[0]})
	feed_dict.update({placeholders['user_vecs']: h[1]})
	feed_dict.update({placeholders['item_vecs']: h[2]})
	feed_dict.update({placeholders['t']: t})
	feed_dict.update({placeholders['batch_index']: b})
	feed_dict.update({placeholders['lr']: learning_rate})
	feed_dict.update({placeholders['mom']: momentum})
	feed_dict.update({placeholders['num_features_nonzero']: h[0][1].shape})
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


#  sample multiple times for each node
def random_walks(adjlist, numerate, walklength):
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


def get_negative_sampling(pairs, adj_nodelist, Q=3, node_sampling='atlas'):
	num_of_nodes = len(adj_nodelist)
	u_i = []
	u_j = []
	graph_label = []
	node_negative_distribution, nodesampling = negative_sampling(adj_nodelist)
	for index in range(0, num_of_nodes):
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
	return u_i, u_j, graph_label


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
