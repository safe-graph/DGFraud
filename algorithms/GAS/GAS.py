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
from base_models.layers import ConcatenationAggregator, AttentionAggregator, GASConcatenation, GraphConvolution


class GAS(keras.Model):
    def __init__(self, args):
        super().__init__()
        self.class_size = args.class_size
        self.reviews_num = args.reviews_num
        self.input_dim_i = args.input_dim_i
        self.input_dim_u = args.input_dim_u
        self.input_dim_r = args.input_dim_r
        self.input_dim_r_gcn = args.input_dim_r_gcn
        self.output_dim1 = args.output_dim1
        self.output_dim2 = args.output_dim2
        self.output_dim3 = args.output_dim3
        self.output_dim4 = args.output_dim4
        self.output_dim5 = args.output_dim5
        self.num_features_nonzero = args.num_features_nonzero
        self.gcn_dim = args.gcn_dim
        self.h_i_size = args.h_i_size
        self.h_u_size = args.h_u_size

        # GAS layers initialization
        self.r_agg_layer = ConcatenationAggregator(input_dim=self.input_dim_r + self.input_dim_u + self.input_dim_i,
                                                   output_dim=self.output_dim1, )

        self.iu_agg_layer = AttentionAggregator(input_dim1=self.h_u_size,
                                                input_dim2=self.h_i_size,
                                                output_dim=self.output_dim3,
                                                hid_dim=self.output_dim2,
                                                concat=True)

        self.r_gcn_layer = GraphConvolution(input_dim=self.input_dim_r_gcn,
                                            output_dim=self.output_dim5,
                                            num_features_nonzero=self.num_features_nonzero,
                                            activation=tf.nn.relu,
                                            dropout=args.dropout,
                                            is_sparse_inputs=True,
                                            norm=True)

        self.concat_layer = GASConcatenation()

        # logistic weights initialization
        self.x_init = tf.keras.initializers.GlorotUniform()
        self.u = tf.Variable(initial_value=self.x_init(
            shape=(self.output_dim1 + 2 * self.output_dim2 + 2 * self.reviews_num + self.output_dim5, self.class_size),
            dtype=tf.float32), trainable=True)

    def __call__(self, inputs):
        support, r_support, features, r_features, label, idx_mask = inputs

        # forward propagation
        h_r = self.r_agg_layer((support, features))
        h_u, h_i = self.iu_agg_layer((support, features))
        outputs = [r_features]
        p_e = self.r_gcn_layer((outputs[-1], r_support), training=True)
        concat_vecs = [h_r, h_u, h_i, p_e]
        gas_out = self.concat_layer((support, concat_vecs))

        # get masked data
        masked_data = tf.gather(gas_out, idx_mask)
        masked_label = tf.gather(label, idx_mask)

        logits = tf.nn.softmax(tf.matmul(masked_data, self.u))
        loss = -tf.reduce_sum(tf.math.log(tf.nn.sigmoid(masked_label * logits)))
        acc = accuracy(logits, masked_label)

        return loss, acc
