'''
This code is due to Yutong Deng (@yutongD), Yingtong Dou (@YingtongDou) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection)
https://github.com/safe-graph/DGFraud

Player2Vec ('Key Player Identification in Underground Forums
over Attributed Heterogeneous Information Network Embedding Framework')

Parameters:
    meta: meta-path number
    nodes: total nodes number
    gcn_output1: the first gcn layer unit number
    gcn_output2: the second gcn layer unit number
    embedding: node feature dim
    encoding: nodes representation dim
'''

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../..')))

import tensorflow as tf
import numpy as np
from tensorflow import keras
from base_models.layers import AttentionLayer, GraphConvolution
from utils.metrics import *


class Player2Vec(keras.Model):
    """
    :param input_dim: the input feature dimension
    :param nhid: the output embedding dimension of the first GCN layer
    :param output_dim: the output embedding dimension of the last GCN layer (number of classes)
    :param args: additional parameters
    """

    def __init__(self, input_dim, nhid, output_dim, args):
        super().__init__()

        self.input_dim = input_dim
        self.nodes = args.nodes
        self.nhid = nhid
        self.class_size = args.class_size
        self.train_size = args.train_size
        self.output_dim = output_dim
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


        # logistic weights initialization
        self.x_init = tf.keras.initializers.GlorotUniform()
        self.u = tf.Variable(initial_value=self.x_init(shape=(self.output_dim, self.class_size), dtype=tf.float32),
                             trainable=True)

    def call(self, inputs, training=True):
        support, x, label, mask = inputs

        outputs = [x]
        for layer in self.layers:
            hidden = layer((outputs[-1], support), training)
            outputs.append(hidden)
        outputs = outputs[-1]

        outputs = tf.reshape(outputs, [1, self.nodes * self.output_dim])
        outputs = AttentionLayer.attention(inputs=outputs, attention_size=1, v_type='tanh')
        outputs = tf.reshape(outputs, [self.nodes, self.output_dim])
        # get masked data
        masked_data = tf.gather(outputs, mask)
        masked_label = tf.gather(label, mask)

        logits = tf.nn.softmax(tf.matmul(masked_data, self.u))
        loss = -tf.reduce_sum(tf.math.log(tf.nn.sigmoid(masked_label * logits)))
        acc = accuracy(logits, masked_label)

        return loss, acc
