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


class GeniePath(Algorithm):

    def __init__(self,
                 session,
                 nodes,
                 in_dim,
                 out_dim,
                 dim,
                 lstm_hidden,
                 heads,
                 layer_num,
                 class_size):
        self.nodes = nodes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dim = dim
        self.lstm_hidden = lstm_hidden
        self.heads = heads
        self.layer_num = layer_num
        self.class_size = class_size

        self.placeholders = {'a': tf.placeholder(tf.float32, [1, self.nodes, self.nodes], 'adj'),
                             'x': tf.placeholder(tf.float32, [self.nodes, self.in_dim], 'nxf'),
                             'batch_index': tf.placeholder(tf.int32, [None], 'index'),
                             't': tf.placeholder(tf.float32, [None, self.out_dim], 'labels'),
                             'lr': tf.placeholder(tf.float32, [], 'learning_rate'),
                             'mom': tf.placeholder(tf.float32, [], 'momentum'),
                             'num_features_nonzero': tf.placeholder(tf.int32)}

        loss, probabilities = self.forward_propagation()
        self.loss, self.probabilities = loss, probabilities
        self.l2 = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(0.01),
                                                         tf.trainable_variables())

        self.pred = tf.one_hot(tf.argmax(self.probabilities, 1), self.out_dim)
        print(self.pred.shape)
        self.correct_prediction = tf.equal(tf.argmax(self.probabilities, 1), tf.argmax(self.placeholders['t'], 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
        print('Forward propagation finished.')

        self.sess = session
        self.optimizer = tf.train.AdamOptimizer(self.placeholders['lr'])
        gradients = self.optimizer.compute_gradients(self.loss + self.l2)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        self.train_op = self.optimizer.apply_gradients(capped_gradients)
        self.init = tf.global_variables_initializer()
        print('Backward propagation finished.')

    def forward_propagation(self):
        with tf.variable_scope('genie_path_forward'):
            x = self.placeholders['x']
            x = x[None, :]
            x = tf.contrib.layers.fully_connected(x, self.dim, activation_fn=lambda x: x)

            gplayers = [GeniePathLayer(self.placeholders, self.nodes, self.in_dim, self.dim)
                        for i in range(self.layer_num)]
            for i, l in enumerate(gplayers):
                x, (h, c) = gplayers[i].forward(x, self.placeholders['a'], self.lstm_hidden, self.lstm_hidden)
                x = x[None, :]
            self.check = x
            x = tf.contrib.layers.fully_connected(x, self.out_dim, activation_fn=lambda x: x)
            x = tf.squeeze(x, 0)
            print('geniePath embedding over!')

        with tf.variable_scope('classification'):
            batch_data = tf.matmul(tf.one_hot(self.placeholders['batch_index'], self.nodes), x)
            # W = tf.get_variable(name='weights',
            #                     shape=[self.out_dim, self.class_size],
            #                     initializer=tf.contrib.layers.xavier_initializer())
            # b = tf.get_variable(name='bias', shape=[1, self.class_size], initializer=tf.zeros_initializer())
            # logits = tf.matmul(batch_data, W) + b
            logits = batch_data
            loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.placeholders['t'], logits=logits)

        return loss, tf.nn.softmax(logits)

    def train(self, x, a, t, b, learning_rate=1e-2, momentum=0.9):
        feed_dict = utils.construct_feed_dict(x, a, t, b, learning_rate, momentum, self.placeholders)
        outs = self.sess.run(
            [self.train_op, self.loss, self.accuracy, self.pred, self.probabilities],
            feed_dict=feed_dict)
        loss = outs[1]
        acc = outs[2]
        pred = outs[3]
        prob = outs[4]
        return loss, acc, pred, prob

    def test(self, x, a, t, b, learning_rate=1e-2, momentum=0.9):
        feed_dict = utils.construct_feed_dict(x, a, t, b, learning_rate, momentum, self.placeholders)
        acc, pred, probabilities, tags = self.sess.run(
            [self.accuracy, self.pred, self.probabilities, self.correct_prediction],
            feed_dict=feed_dict)
        return acc, pred, probabilities, tags
