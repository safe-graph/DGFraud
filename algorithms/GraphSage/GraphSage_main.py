"""
This code is due to Yutong Deng (@yutongD), Yingtong Dou (@YingtongDou) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection)
https://github.com/safe-graph/DGFraud
"""
import os
import sys
import argparse
from tqdm import tqdm
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../..')))

import numpy as np
np.seterr(divide='ignore',invalid='ignore')
import tensorflow as tf
import collections
from sklearn.metrics import f1_score, accuracy_score

from GraphSage import GraphSage
from utils.data_loader import *
from utils.utils import *

# init the common args, expect the model specific args
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=717, help='random seed')
parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--train_size', type=float, default=0.8, help='training set percentage')
parser.add_argument('--lr', type=float, default=0.5, help='learning rate')
parser.add_argument('--nhid', type=int, default=128, help='number of hidden units in GCN')
parser.add_argument('--sample_sizes', type=list, default=[5, 5], help='number of samples for each layer')
args = parser.parse_args()

# set seed
np.random.seed(args.seed)
tf.random.set_seed(args.seed)

def main(neigh_dict: dict, features: np.ndarray, labels: np.ndarray, masks: list, num_classes: int, args):
	train_nodes = masks[0]
	val_nodes = masks[1]
	test_nodes = masks[2]

	# training
	def generate_training_minibatch(nodes_for_training, all_labels, batch_size):
		nodes_for_epoch = np.copy(nodes_for_training)
		ix = 0
		np.random.shuffle(nodes_for_epoch)
		while len(nodes_for_epoch) > ix + batch_size:
			mini_batch_nodes = nodes_for_epoch[ix:ix+batch_size]
			batch = build_batch(mini_batch_nodes, neigh_dict, args.sample_sizes)
			labels = all_labels[mini_batch_nodes]
			ix += batch_size
			yield (batch, labels)
		mini_batch_nodes = nodes_for_epoch[ix:-1]
		batch = build_batch(mini_batch_nodes, neigh_dict, args.sample_sizes)
		labels = all_labels[mini_batch_nodes]
		yield (batch, labels)

	model = GraphSage(features, args.nhid, len(args.sample_sizes), num_classes)
	optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr)
	loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

	for epoch in range(args.epochs):
		print(f"Epoch {epoch:d}: training...")
		minibatch_generator = generate_training_minibatch(train_nodes, labels, args.batch_size)
		for inputs, inputs_labels in tqdm(minibatch_generator, total=len(train_nodes) / args.batch_size):
			with tf.GradientTape() as tape:
				predicted = model(inputs)
				loss = loss_fn(tf.convert_to_tensor(inputs_labels), predicted)
				acc = accuracy_score(inputs_labels, predicted.numpy().argmax(axis=1))
			grads = tape.gradient(loss, model.trainable_weights)
			optimizer.apply_gradients(zip(grads, model.trainable_weights))
			print(f" loss: {loss.numpy():.4f}, acc: {acc:.4f}")
		
		# validation
		print("Validating...")
		val_results = model(build_batch(val_nodes, neigh_dict, args.sample_sizes))
		loss = loss_fn(tf.convert_to_tensor(labels[val_nodes]), val_results)
		val_acc = accuracy_score(labels[val_nodes], val_results.numpy().argmax(axis=1))
		print(f" Epoch: {epoch:d}, loss: {loss.numpy():.4f}, acc: {acc:.4f}")

	# testing
	print("Testing...")
	results = model(build_batch(test_nodes, neigh_dict, args.sample_sizes))
	# score = f1_score(labels[test_nodes], results.numpy().argmax(axis=1), average="micro")
	test_acc = accuracy_score(labels[test_nodes], results.numpy().argmax(axis=1))
	print(f"Test acc: {test_acc:.4f}")


def build_batch(nodes, neigh_dict, sample_sizes):
	"""
	:param [int] nodes: node ids
	:param {node:[node]} neigh_dict: BIDIRECTIONAL adjacency matrix in dict
	:param [sample_size]: sample sizes for each layer, lens is the number of layers
	:return namedtuple minibatch
		"src_nodes": node ids to retrieve from raw feature and feed to the first layer
		"dstsrc2srcs": list of dstsrc2src matrices from last to first layer
		"dstsrc2dsts": list of dstsrc2dst matrices from last to first layer
		"dif_mats": list of dif_mat matrices from last to first layer
	"""
		
	dst_nodes = [nodes]
	dstsrc2dsts = []
	dstsrc2srcs = []
	dif_mats = []

	max_node_id = max(list(neigh_dict.keys()))

	for sample_size in reversed(sample_sizes):
		ds, d2s, d2d, dm = _compute_diffusion_matrix(dst_nodes[-1],
													neigh_dict,
													sample_size,
													max_node_id,
													)
		dst_nodes.append(ds)
		dstsrc2srcs.append(d2s)
		dstsrc2dsts.append(d2d)
		dif_mats.append(dm)

	src_nodes = dst_nodes.pop()
		
	MiniBatchFields = ["src_nodes", "dstsrc2srcs", "dstsrc2dsts", "dif_mats"]
	MiniBatch = collections.namedtuple ("MiniBatch", MiniBatchFields)

	return MiniBatch(src_nodes, dstsrc2srcs, dstsrc2dsts, dif_mats)


def _compute_diffusion_matrix(dst_nodes, neigh_dict, sample_size, max_node_id):

	def sample(ns):
		return np.random.choice(ns, min(len(ns), sample_size), replace=False)

	def vectorize(ns):
		v = np.zeros(max_node_id + 1, dtype=np.float32)
		v[ns] = 1
		return v

	# sample neighbors
	adj_mat_full = np.stack([vectorize(sample(neigh_dict[n])) for n in dst_nodes])
	nonzero_cols_mask = np.any(adj_mat_full.astype(np.bool), axis=0)

	# compute diffusion matrix
	adj_mat = adj_mat_full[:, nonzero_cols_mask]
	adj_mat_sum = np.sum(adj_mat, axis=1, keepdims=True)
	dif_mat = np.nan_to_num(adj_mat / adj_mat_sum)

	# compute dstsrc mappings
	src_nodes = np.arange(nonzero_cols_mask.size)[nonzero_cols_mask]
	# np.union1d automatic sorts the return, which is required for np.searchsorted
	dstsrc = np.union1d(dst_nodes, src_nodes)
	dstsrc2src = np.searchsorted(dstsrc, src_nodes)
	dstsrc2dst = np.searchsorted(dstsrc, dst_nodes)

	return dstsrc, dstsrc2src, dstsrc2dst, dif_mat


if __name__ == "__main__":
	# load the data
	adj_list, features, idx_train, _, idx_val, _, idx_test, _, y = load_data_yelp(meta=False, train_size=args.train_size)

	num_classes = len(set(y))
	label = np.array([y]).T

	features = normalize_feature(features)
	features = np.array(features.todense())

	neigh_dict = collections.defaultdict(list)
	for i in range(len(y)):
		neigh_dict[i] = []
	for net in adj_list:
		nodes1 = net.nonzero()[0]
		nodes2 = net.nonzero()[1]
		for node1, node2 in zip(nodes1, nodes2):
			neigh_dict[node1].append(node2)

	neigh_dict = {k: np.array(v, dtype=np.int64) for k, v in neigh_dict.items()}

	main(neigh_dict, features, label, [idx_train, idx_val, idx_test], num_classes, args)
