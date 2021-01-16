"""
This code is due to Yutong Deng (@yutongD), Yingtong Dou (@YingtongDou) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection)
https://github.com/safe-graph/DGFraud

FdGars ('FdGars: Fraudster Detection via Graph Convolutional Networks in Online App Review System')
"""


import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../..')))

import tensorflow as tf
from tensorflow import keras
from base_models.layers import GraphConvolution
from utils.metrics import *


class FdGars(keras.Model):
	"""
	:param input_dim: the input feature dimension
	:param nhid: the output embedding dimension of the first GCN layer
	:param output_dim: the output embedding dimension of the last GCN layer (number of classes)
	:param args: additional parameters
	"""
	def __init__(self, input_dim, nhid, output_dim, args):
		super().__init__()

		self.input_dim = input_dim
		self.nhid = nhid
		self.output_dim = output_dim
		self.weight_decay = args.weight_decay
		self.num_features_nonzero = args.num_features_nonzero

		self.layers_ = []
		self.layers_.append(GraphConvolution(input_dim=self.input_dim,
											output_dim=self.nhid,
											num_features_nonzero=self.num_features_nonzero,
											activation=tf.nn.relu,
											dropout=args.dropout,
											is_sparse_inputs=True,
											norm=True))

		self.layers_.append(GraphConvolution(input_dim=self.nhid,
											output_dim=self.output_dim,
											num_features_nonzero=self.num_features_nonzero,
											activation=lambda x: x,
											dropout=args.dropout,
											norm=False))

	def call(self, inputs, training=True):

		support, x, label, mask = inputs

		outputs = [x]

		# forward propagation
		for layer in self.layers:
			hidden = layer((outputs[-1], support), training)
			outputs.append(hidden)
		output = outputs[-1]

		# Weight decay loss
		loss = tf.zeros([])
		for var in self.layers_[0].trainable_variables:
			loss += self.weight_decay * tf.nn.l2_loss(var)

		# Cross entropy loss
		loss += masked_softmax_cross_entropy(output, label, mask)

		# Prediction results
		acc = masked_accuracy(output, label, mask)

		return loss, acc
