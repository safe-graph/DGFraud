'''
This code is due to Zhiwei Liu (@JimLiu96) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection)
https://github.com/safe-graph/DGFraud
'''
from __future__ import division
from __future__ import print_function

from layers import Layer

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS


"""
Classes that are used to sample node neighborhoods
"""

class UniformNeighborSampler(Layer):
    """
    Uniformly samples neighbors.
    Assumes that adj lists are padded with random re-sampling
    """
    def __init__(self, adj_info, **kwargs):
        super(UniformNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info

    def _call(self, inputs):
        ids, num_samples = inputs
        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids) 
        adj_lists = tf.transpose(tf.random_shuffle(tf.transpose(adj_lists)))
        adj_lists = tf.slice(adj_lists, [0,0], [-1, num_samples])
        return adj_lists

class DistanceNeighborSampler(Layer):
    """
    Sampling neighbors based on the feature consistency.
    """
    def __init__(self, adj_info, **kwargs):
        super(DistanceNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info
        self.num_neighs = adj_info.shape[-1]

    def _call(self, inputs):
        eps = 0.001
        ids, num_samples, features, batch_size = inputs
        adj_lists = tf.gather(self.adj_info, ids)
        node_features = tf.gather(features, ids)
        feature_size = tf.shape(features)[-1]
        node_feature_repeat = tf.tile(node_features, [1,self.num_neighs])
        node_feature_repeat = tf.reshape(node_feature_repeat, [batch_size, self.num_neighs, feature_size])
        neighbor_feature =  tf.gather(features, adj_lists)
        distance = tf.sqrt(tf.reduce_sum(tf.square(node_feature_repeat - neighbor_feature), -1))
        prob = tf.exp(-distance)
        prob_sum = tf.reduce_sum(prob, -1, keepdims=True)
        prob_sum = tf.tile(prob_sum, [1,self.num_neighs])
        prob = tf.divide(prob, prob_sum)
        prob = tf.where(prob>eps, prob, 0*prob) # uncommenting this line to use eps to filter small probabilities
        samples_idx = tf.random.categorical(tf.math.log(prob), num_samples)
        selected = tf.batch_gather(adj_lists, samples_idx)
        return selected


