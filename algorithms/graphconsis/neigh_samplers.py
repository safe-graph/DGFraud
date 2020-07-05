from __future__ import division
from __future__ import print_function

from graphconsis.layers import Layer

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
        # print('!!!!!!!!!neigh_samplers adj info shape',self.adj_info.shape)
        self.num_neighs = adj_info.shape[-1]

    def _call(self, inputs):
        ids, num_samples, features, batch_size = inputs
        # adj_lists = tf.nn.embedding_lookup(self.adj_info, ids)
        adj_lists = tf.gather(self.adj_info, ids)
        # print('!!!!!!!!neigh_samplers,return adj_lists shape',adj_lists.shape)
        # print('!!!!!!!!neigh_samplers, print ids shape',ids.shape)
        # for i in ids:
        # node_features = tf.nn.embedding_lookup(features, ids)
        node_features = tf.gather(features, ids)
        # num_neighs = tf.shape(self.adj_info)[-1]
        feature_size = tf.shape(features)[-1]
        # print('!!!!!!!!neigh_samplers:batch_size:',batch_size, self.num_neighs, feature_size)
        node_feature_repeat = tf.tile(node_features, [1,self.num_neighs])
        node_feature_repeat = tf.reshape(node_feature_repeat, [batch_size, self.num_neighs, feature_size])
        neighbor_feature =  tf.gather(features, adj_lists)
        distance = tf.sqrt(tf.reduce_sum(tf.square(node_feature_repeat - neighbor_feature), -1))
        prob = tf.exp(-distance)
        prob_sum = tf.reduce_sum(prob, -1, keepdims=True)
        prob_sum = tf.tile(prob_sum, [1,self.num_neighs])
        prob = tf.divide(prob, prob_sum)
        samples_idx = tf.random.categorical(tf.math.log(prob), num_samples)
        selected = tf.batch_gather(adj_lists, samples_idx)
        # print('!!!!!!!!neigh_samplers,return samples shape', selected.shape)
        return selected

    # def distance(self, x, Y, dis='l2'):
    #     repeat_num = Y.size()[0]
    #     X = x.repeat(repeat_num, 1)
    #     d = tf.diag((X-Y).matmul(tf.transpose(X-Y)))
    #     if dis == 'l2_sq':
    #         return d
    #     if dis == 'l2':
    #         return tf.sqrt(d)

#     def dis2prob(self, distance, prob_thres=0.01, smooth=0.001):
#         prob = tf.exp(-distance)
#         prob += smooth
#         probNorm = prob / (distance.norm(2, dim=0) + smooth)
# #         probNorm[probNorm < prob_thres] = 0.0
#         probNorm = probNorm/sum(probNorm)
#         return probNorm
