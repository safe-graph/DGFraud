"""
This code is due to Yutong Deng (@yutongD), Yingtong Dou (@YingtongDou) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection)
https://github.com/safe-graph/DGFraud
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../..')))

import tensorflow as tf
from tensorflow.keras import optimizers

import argparse
from tqdm import tqdm

from algorithms.FdGars.FdGars import FdGars

from utils.data_loader import *
from utils.utils import *

# init the common args, expect the model specific args
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--train_size', type=float, default=0.2, help='training set percentage')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--nhid', type=int, default=128, help='number of hidden units in GCN')

args = parser.parse_args()

# set seed
np.random.seed(args.seed)
tf.random.set_seed(args.seed)


def main(support: list, features: tf.SparseTensor, label: tf.Tensor, masks: list, args):
	"""
	Main function to train, val and test the model
	:param support: a list of the sparse adjacency matrix
	:param features: the feature sparse tensor for all nodes
	:param label: the label tensor for all nodes
	:param masks: a list of mask tensors to obtain the train, val, and test data
	:param args: additional parameters
	"""
	model = FdGars(args.input_dim, args.nhid, args.output_dim, args)
	optimizer = optimizers.Adam(lr=args.lr)

	# train
	for epoch in tqdm(range(args.epochs)):

		with tf.GradientTape() as tape:
			train_loss, train_acc = model([support, features, label, masks[0]])

		grads = tape.gradient(train_loss, model.trainable_variables)
		optimizer.apply_gradients(zip(grads, model.trainable_variables))

		val_loss, val_acc = model([support, features, label, masks[1]], training=False)

		if epoch % 10 == 0:
			print(f"Epoch: {epoch:d}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f},"
				  f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

	# test
	_, test_acc = model([support, features, label, masks[2]], training=False)
	print(f"Test acc: {test_acc:.4f}")


if __name__ == "__main__":

	# load the data
	adj_list, features, idx_train, _, idx_val, _, idx_test, _, y = load_data_dblp(meta=False, train_size=args.train_size)

	# convert to dense tensors
	train_mask = tf.convert_to_tensor(sample_mask(idx_train, y.shape[0]))
	val_mask = tf.convert_to_tensor(sample_mask(idx_val, y.shape[0]))
	test_mask = tf.convert_to_tensor(sample_mask(idx_test, y.shape[0]))
	label = tf.convert_to_tensor(y)

	# get sparse tuples
	features = preprocess_feature(features)
	support = preprocess_adj(adj_list[0])

	# initialize the model parameters
	args.input_dim = features[2][1]
	args.output_dim = y.shape[1]
	args.train_size = len(idx_train)
	args.num_features_nonzero = features[1].shape

	# get sparse tensors
	features = tf.SparseTensor(*features)
	support = [tf.cast(tf.SparseTensor(*support), dtype=tf.float32)]

	main(support, features, label, [train_mask, val_mask, test_mask], args)
