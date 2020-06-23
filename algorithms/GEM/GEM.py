'''
This code is due to Yutong Deng (@yutongD), Yingtong Dou (@YingtongDou) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection)
https://github.com/safe-graph/DGFraud

GEM ('Heterogeneous Graph Neural Networks for Malicious Account Detection')

Parameters:
    nodes: total nodes number
    meta: device number
    hop:  the number of hops a vertex needs to look at, or the number of hidden layers
    embedding: node feature dim
    encoding: nodes representation dim
'''
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../..')))
import tensorflow as tf
from base_models.models import GEMLayer
from algorithms.base_algorithm import Algorithm
from utils import utils


class GEM(Algorithm):

    def __init__(self,
                 session,
                 nodes,
                 class_size,
                 meta,
                 embedding,
                 encoding,
                 hop):
        self.nodes = nodes
        self.meta = meta
        self.class_size = class_size
        self.embedding = embedding
        self.encoding = encoding
        self.hop = hop

        self.placeholders = {'a': tf.placeholder(tf.float32, [self.meta, self.nodes, self.nodes], 'adj'),
                             'x': tf.placeholder(tf.float32, [self.nodes, self.embedding], 'nxf'),
                             'batch_index': tf.placeholder(tf.int32, [None], 'index'),
                             't': tf.placeholder(tf.float32, [None, self.class_size], 'labels'),
                             'lr': tf.placeholder(tf.float32, [], 'learning_rate'),
                             'mom': tf.placeholder(tf.float32, [], 'momentum'),
                             'num_features_nonzero': tf.placeholder(tf.int32)}

        loss, probabilities = self.forward_propagation()
        self.loss, self.probabilities = loss, probabilities
        self.l2 = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(0.01),
                                                         tf.trainable_variables())

        x = tf.ones_like(self.probabilities)
        y = tf.zeros_like(self.probabilities)
        self.pred = tf.where(self.probabilities > 0.5, x=x, y=y)

        print(self.pred.shape)
        self.correct_prediction = tf.equal(self.pred, self.placeholders['t'])
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
        with tf.variable_scope('gem_embedding'):
            h = tf.get_variable(name='init_embedding', shape=[self.nodes, self.encoding],
                                initializer=tf.contrib.layers.xavier_initializer())
            for i in range(0, self.hop):
                f = GEMLayer(self.placeholders, self.nodes, self.meta, self.embedding, self.encoding)
                gem_out = f(inputs=h)
                h = tf.reshape(gem_out, [self.nodes, self.encoding])
            print('GEM embedding over!')

        with tf.variable_scope('classification'):
            batch_data = tf.matmul(tf.one_hot(self.placeholders['batch_index'], self.nodes), h)
            W = tf.get_variable(name='weights',
                                shape=[self.encoding, self.class_size],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name='bias', shape=[1, self.class_size], initializer=tf.zeros_initializer())
            tf.transpose(batch_data, perm=[0, 1])
            logits = tf.matmul(batch_data, W) + b

            u = tf.get_variable(name='u',
                                shape=[1, self.encoding],
                                initializer=tf.contrib.layers.xavier_initializer())

            loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.placeholders['t'], logits=logits)

            # TODO
            # loss = -tf.reduce_sum(
            #     tf.log_sigmoid(self.placeholders['t'] * tf.matmul(u, tf.transpose(batch_data, perm=[1, 0]))))

        # return loss, logits
        return loss, tf.nn.sigmoid(logits)

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
