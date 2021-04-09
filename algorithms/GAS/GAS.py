'''
This code is due to Yutong Deng (@yutongD), Yingtong Dou (@YingtongDou) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection)
https://github.com/safe-graph/DGFraud
GAS ('Spam Review Detection with Graph Convolutional Networks')
Parameters:
	nodes: total nodes number
	class_size: class number
	embedding_i: item embedding size
	embedding_u: user embedding size
	embedding_r: review embedding size
	gcn_dim: the gcn layer unit number
'''
import os
import sys
from utils.metrics import accuracy

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../..')))
import tensorflow as tf
from tensorflow import keras
from base_models.layers import ConcatenationAggregator, AttentionAggregator, GASConcatenation


class GAS(keras.Model):
	def __init__(self, input_dim_i, input_dim_u, input_dim_r, h_u_size, h_i_size,
				 output_dim1, output_dim2, output_dim3, output_dim4, gcn_dim, args):
		super().__init__()
		self.class_size = args.class_size
		self.nodes_num = args.nodes_num
		self.input_dim_i = input_dim_i
		self.input_dim_u = input_dim_u
		self.input_dim_r = input_dim_r
		self.output_dim1 = output_dim1
		self.output_dim2 = output_dim2
		self.output_dim3 = output_dim3
		self.output_dim4 = output_dim4
		self.gcn_dim = gcn_dim
		self.h_i_size = h_i_size
		self.h_u_size = h_u_size

		self.x_init = tf.keras.initializers.GlorotUniform()
		self.u = tf.Variable(initial_value=self.x_init(
			shape=(self.output_dim1 + 2 * self.output_dim2 + 2 * self.nodes_num + self.nodes_num, self.class_size),
			dtype=tf.float32), trainable=True)
		# GAS layers initialization
		# dou: self.input_dim_u, self.imput_dim_i
		# dou: figure out whether using the is_sparse_input argument for three following layers
		self.r_agg_layer = ConcatenationAggregator(input_dim=self.input_dim_r + input_dim_u + input_dim_i,
												   output_dim=self.output_dim1,
												   is_sparse_inputs=True)

		self.iu_agg_layer = AttentionAggregator(input_dim1=self.h_u_size,
												input_dim2=self.h_i_size,
												output_dim=self.output_dim3,
												hid_dim=self.output_dim2,
												is_sparse_inputs=True,
												concat=True)

		self.concat_layer = GASConcatenation(is_sparse_inputs=True)

		# dou: add a GCN layer to pass adj_list[6] and review features to get the $p_e$ in the paper
		# which will replace the adj_list[6] in the GASConcatenation layer
	def __call__(self, inputs):
		supports, x, label, idx_mask = inputs

		# forward propagation
		h_r = self.r_agg_layer((supports, x))
		h_u, h_i = self.iu_agg_layer((supports, x))
		concat_vecs = [h_r, h_u, h_i]
		gas_out = self.concat_layer((supports, concat_vecs))

		# get masked data
		masked_data = tf.gather(gas_out, idx_mask)
		masked_label = tf.gather(label, idx_mask)

		logits = tf.nn.softmax(tf.matmul(masked_data, self.u))
		loss = -tf.reduce_sum(tf.math.log(tf.nn.sigmoid(masked_label * logits)))
		acc = accuracy(logits, masked_label)

		return loss, acc

