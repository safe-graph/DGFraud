'''
This code is due to Yutong Deng (@yutongD), Yingtong Dou (@YingtongDou) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection)
https://github.com/safe-graph/DGFraud

GeniePath ('GeniePath: Graph Neural Networks with Adaptive Receptive Paths')

Parameters:
    nodes: total nodes number
    in_dim: input feature dim
    out_dim: output representation dim
    dim: breadth forward layer unit
    lstm_hidden: depth forward layer unit
    layer_num: GeniePath layer num
'''
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../..')))
import tensorflow as tf
from base_models.layers import GeniePathLayer
from algorithms.base_algorithm import Algorithm
from utils import utils
from utils.metrics import *


class GeniePath(Algorithm):

    def __init__(self, input_dim, output_dim, args):
        super().__init__()

        self.nodes_num = args.nodes_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.GAT_output_dim = args.GAT_output_dim
        self.lstm_hidden = args.lstm_hidden
        self.GAT_heads = args.GAT_heads
        self.layer_num = args.layer_num
        self.class_size = args.class_size
        # self.trainable_variables = []

        # Fully connection initialization
        self.Fully_connected_net_input = tf.keras.layers.Dense(self.GAT_output_dim, activation=lambda x: x)
        self.Fully_connected_net_output = tf.keras.layers.Dense(self.output_dim, activation=lambda x: x)

        # GeniePath layers initialization
        self.layers_ = []
        for _ in range(self.layer_num):
            self.layers_.append(GeniePathLayer(self.nodes_num, self.input_dim, self.GAT_output_dim, self.lstm_hidden, self.GAT_heads))


    def __call__(self, inputs):
        """
        @param inputs: include support, x, label, mask
        support: a list of the sparse adjacency Tensor
        x: the node feature
        label: the label tensor
        mask: a list of mask tensors to obtain the data index
        @return:
        """

        supports, x, label, idx_mask = inputs

        # forward propagation
        x = tf.expand_dims(x, axis=0)
        x = self.Fully_connected_net_input(x)
        # self.trainable_variables.append(self.Fully_connected_net_input.trainable_variables)
        for layer in self.layers_:
            x = layer((x, supports))
            # self.trainable_variables.append(layer.trainable_variables)
            x = tf.expand_dims(x, axis=0)
        x = self.Fully_connected_net_output(x)
        # self.trainable_variables.append(self.Fully_connected_net_output.trainable_variables)
        # self.trainable_variables = sum(self.trainable_variables, [])
        GeniePath_out = tf.squeeze(x,0)

        # get masked data
        masked_data = tf.gather(GeniePath_out, idx_mask)
        masked_label = tf.gather(label, idx_mask)

        # loss and accuracy
        logits = masked_data
        loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(masked_label, logits))
        acc = accuracy(logits, masked_label)

        return loss, acc


