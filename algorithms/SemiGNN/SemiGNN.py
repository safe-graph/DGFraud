'''
This code is due to Yutong Deng (@yutongD), Yingtong Dou (@Yingtong Dou) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection)
https://github.com/safe-graph/DGFraud

SemiGNN ('A Semi-supervised Graph Attentive Network for
        Financial Fraud Detection')

Parameters:
    nodes: total nodes number
    semi_encoding1: node attention layer unit number
    semi_encoding2: view attention layer unit number
    semi_encoding3: MLP layer unit number
    init_emb_size: the initial node embedding
    meta: view number
    gcn_output1: the first gcn layer unit number
    gcn_output2: the second gcn layer unit number
    ul: labeled users number
'''

import tensorflow as tf
from algorithms.base_algorithm import Algorithm
from base_models.layers import AttentionLayer
from utils import utils


class SemiGNN(Algorithm):

    def __init__(self,
                 session,
                 nodes,
                 class_size,
                 semi_encoding1,
                 semi_encoding2,
                 semi_encoding3,
                 init_emb_size,
                 meta,
                 ul,
                 alpha,
                 lamtha):
        self.nodes = nodes
        self.meta = meta
        self.class_size = class_size
        self.semi_encoding1 = semi_encoding1
        self.semi_encoding2 = semi_encoding2
        self.semi_encoding3 = semi_encoding3
        self.init_emb_size = init_emb_size
        self.ul = ul
        self.alpha = alpha
        self.lamtha = lamtha
        self.placeholders = {'a': tf.placeholder(tf.float32, [self.meta, self.nodes, None], 'adj'),
                             'u_i': tf.placeholder(tf.float32, [None, ], 'u_i'),
                             'u_j': tf.placeholder(tf.float32, [None, ], 'u_j'),
                             'batch_index': tf.placeholder(tf.int32, [None], 'index'),
                             'sup_t': tf.placeholder(tf.float32, [None, self.class_size], 'sup_t'),
                             'graph_t': tf.placeholder(tf.float32, [None, 1], 'graph_t'),
                             'lr': tf.placeholder(tf.float32, [], 'learning_rate'),
                             'mom': tf.placeholder(tf.float32, [], 'momentum'),
                             'num_features_nonzero': tf.placeholder(tf.int32)}

        loss, probabilities, pred = self.forward_propagation()
        self.loss, self.probabilities, self.pred = loss, probabilities, pred
        self.l2 = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(0.01),
                                                         tf.trainable_variables())

        print(self.pred.shape)
        self.correct_prediction = tf.equal(tf.argmax(self.probabilities, 1), tf.argmax(self.placeholders['sup_t'], 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
        print('Forward propagation finished.')

        self.sess = session
        self.optimizer = tf.train.AdamOptimizer(self.placeholders['lr'])
        gradients = self.optimizer.compute_gradients(self.loss + self.lamtha * self.l2)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        self.train_op = self.optimizer.apply_gradients(capped_gradients)
        self.init = tf.global_variables_initializer()
        print('Backward propagation finished.')

    def forward_propagation(self):
        with tf.variable_scope('node_level_attention', reuse=tf.AUTO_REUSE):
            h1 = []
            for i in range(self.meta):
                emb = tf.get_variable(name='init_embedding', shape=[self.nodes, self.nodes, self.init_emb_size],
                                      initializer=tf.contrib.layers.xavier_initializer())
                h = AttentionLayer.node_attention(inputs=emb, encoding=self.semi_encoding1)
                h = tf.reshape(h, [self.nodes, self.semi_encoding1])
                h1.append(h)
            h1 = tf.concat(h1, 0)
            h1 = tf.reshape(h1, [self.meta, self.nodes * self.semi_encoding1])
            print('Node_level attention over!')

        with tf.variable_scope('view_level_attention'):
            h2 = AttentionLayer.view_attention(inputs=h1, encoding=(self.semi_encoding2 * self.nodes), meta=self.meta)
            h2 = tf.reshape(h2, [self.nodes, self.semi_encoding2 * self.meta])
            print('View_level attention over!')

        with tf.variable_scope('MLP'):
            W1 = tf.get_variable(name='weights_1', shape=[self.semi_encoding2 * self.meta, self.semi_encoding3],
                                 initializer=tf.contrib.layers.xavier_initializer())
            h3 = tf.matmul(h2, W1)  # pair

        with tf.variable_scope('loss'):
            batch_data = tf.matmul(tf.one_hot(self.placeholders['batch_index'], self.nodes), h3)
            W2 = tf.get_variable(name='weights_2', shape=[self.semi_encoding3, self.class_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name='bias', shape=[1, self.class_size], initializer=tf.zeros_initializer())
            tf.transpose(batch_data, perm=[0, 1])
            logits = tf.matmul(batch_data, W2) + b
            prob = tf.nn.sigmoid(logits)
            pred = tf.one_hot(tf.argmax(prob, 1), self.class_size)

            flag = (tf.cast(tf.reduce_sum(
                tf.cast(tf.equal(self.placeholders['sup_t'], pred), dtype=tf.int32), 1), dtype=tf.bool))
            flag = tf.expand_dims(tf.cast(flag, tf.float32), 1)
            loss1 = -(1 / self.ul) * tf.reduce_sum(
                flag * tf.log(tf.nn.softmax(logits)))

            u_i_embedding = tf.nn.embedding_lookup(h3, tf.cast(self.placeholders['u_i'], dtype=tf.int32))
            u_j_embedding = tf.nn.embedding_lookup(h3, tf.cast(self.placeholders['u_j'], dtype=tf.int32))
            inner_product = tf.reduce_sum(u_i_embedding * u_j_embedding, axis=1)
            loss2 = -tf.reduce_mean(tf.log_sigmoid(self.placeholders['graph_t'] * inner_product))

            loss = self.alpha * loss1 + (1 - self.alpha) * loss2
        return loss, prob, pred

    def train(self, a, u_i, u_j, batch_graph_label, batch_data, batch_sup_label, learning_rate=1e-2, momentum=0.9):
        feed_dict = utils.construct_feed_dict_semi(a, u_i, u_j, batch_graph_label, batch_data, batch_sup_label,
                                                   learning_rate, momentum,
                                                   self.placeholders)
        outs = self.sess.run(
            [self.train_op, self.loss, self.accuracy, self.pred, self.probabilities],
            feed_dict=feed_dict)
        loss = outs[1]
        acc = outs[2]
        pred = outs[3]
        prob = outs[4]
        return loss, acc, pred, prob

    def test(self, a, u_i, u_j, batch_graph_label, batch_data, batch_sup_label, learning_rate=1e-2, momentum=0.9):
        feed_dict = utils.construct_feed_dict_semi(a, u_i, u_j, batch_graph_label, batch_data, batch_sup_label,
                                                   learning_rate, momentum,
                                                   self.placeholders)
        acc, pred, probabilities, tags = self.sess.run(
            [self.accuracy, self.pred, self.probabilities, self.correct_prediction],
            feed_dict=feed_dict)
        return acc, pred, probabilities, tags
