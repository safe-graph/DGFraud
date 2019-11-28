# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

from models.model import GCN, GAT
import scipy.sparse as sp
import os
import time

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
tf.reset_default_graph()
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

data_size = 9067
node_size = 9067
node_embedding = 100  # feature size
node_encoding = 100  # gcn embedding size
meta_size = 3  # number of metapath

# GCN layer unit
gcn_para = [16, 16]

# train
batch_size = 64
epoch_num = 1
learning_rate = 0.01
momentum = 0.9


def read_data():
    index = list(range(9067))
    y = np.loadtxt('data/label.txt')
    X_train, X_test, y_train, y_test = train_test_split(index, y, stratify=y, test_size=0.4,
                                                        random_state=48, shuffle=True)

    return X_train, y_train, X_test, y_test


train_data, train_label, test_data, test_label = read_data()
train_size = len(train_data)


# get batch data
def get_data(ix, int_batch):
    if ix + int_batch >= train_size:
        ix = train_size - int_batch
        end = train_size
    else:
        end = ix + int_batch
    return train_data[ix:end], train_label[ix:end]


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).A


class Player2Vec(object):
    def __init__(self, session,
                 meta,
                 nodes,
                 class_size,
                 gcn_output1,
                 gcn_output2,
                 embedding,
                 encoding):
        self.meta = meta
        self.nodes = nodes
        self.class_size = class_size
        self.gcn_output1 = gcn_output1
        self.gcn_output2 = gcn_output2
        self.embedding = embedding
        self.encoding_size = encoding

        self.build_placeholders()

        loss, probabilities, features = self.forward_propagation()
        self.loss, self.probabilities, self.features = loss, probabilities, features
        self.l2 = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(0.01),
                                                         tf.trainable_variables())

        self.pred = tf.one_hot(tf.argmax(self.probabilities, 1), class_size)
        print(self.pred.shape)
        self.correct_prediction = tf.equal(tf.argmax(self.probabilities, 1), tf.argmax(self.t, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
        print('Forward propagation finished.')

        self.sess = session
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        gradients = self.optimizer.compute_gradients(self.loss + self.l2)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        self.train_op = self.optimizer.apply_gradients(capped_gradients)
        self.init = tf.global_variables_initializer()
        print('Backward propagation finished.')

    def build_placeholders(self):
        self.a = tf.placeholder(tf.float32, [self.meta, self.nodes, self.nodes], 'adj')
        self.x = tf.placeholder(tf.float32, [self.nodes, self.embedding], 'nxf')
        self.batch_index = tf.placeholder(tf.int32, [None], 'index')
        self.t = tf.placeholder(tf.float32, [None, self.class_size], 'labels')
        self.lr = tf.placeholder(tf.float32, [], 'learning_rate')
        self.mom = tf.placeholder(tf.float32, [], 'momentum')

    def forward_propagation(self):
        with tf.variable_scope('gcn'):
            x = self.x
            A = tf.reshape(self.a, [self.meta, self.nodes, self.nodes])
            gcn_emb = []
            for i in range(self.meta):
                gcn_out = tf.reshape(GCN(x, A[i], gcn_para[0], gcn_para[1], node_embedding,
                                         node_encoding).embedding(), [1, self.nodes * self.encoding_size])
                gcn_emb.append(gcn_out)
            gcn_emb = tf.concat(gcn_emb, 0)
            assert gcn_emb.shape == [self.meta, self.nodes * self.embedding]
            print('GCN embedding over!')

        with tf.variable_scope('gat'):
            x = gcn_out
            x = tf.expand_dims(x, 0)
            gat_out = GAT.attention(inputs=x, attention_size=1)
            gat_out = tf.reshape(gat_out, [self.nodes, self.encoding_size])
            print('GAT embedding over!')

        with tf.variable_scope('classification'):
            batch_data = tf.matmul(tf.one_hot(self.batch_index, self.nodes), gat_out)
            W = tf.get_variable(name='weights', shape=[self.encoding_size, self.class_size],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name='bias', shape=[1, self.class_size], initializer=tf.zeros_initializer())
            tf.transpose(batch_data, perm=[0, 1])
            logits = tf.matmul(batch_data, W) + b
            loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.t, logits=logits)

        return loss, tf.nn.sigmoid(logits), gcn_out[0]

    def train(self, x, a, t, b, learning_rate=1e-2, momentum=0.9):
        feed_dict = {
            self.x: x,
            self.a: a,
            self.t: t,
            self.batch_index: b,
            self.lr: learning_rate,
            self.mom: momentum
        }
        outs = self.sess.run(
            [self.train_op, self.loss, self.accuracy, self.pred, self.probabilities],
            feed_dict=feed_dict)
        loss = outs[1]
        acc = outs[2]
        pred = outs[3]
        prob = outs[4]
        return loss, acc, pred, prob

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver()
        save_path = saver.save(sess, "tmp/%s.ckpt" % 'temp')
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver()
        save_path = "tmp/%s.ckpt" % 'temp'
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)

    def test(self, x, a, t, b):
        feed_dict = {
            self.x: x,
            self.a: a,
            self.t: t,
            self.batch_index: b
        }
        acc, pred, features, probabilities, tags = self.sess.run(
            [self.accuracy, self.pred, self.features, self.probabilities, self.correct_prediction],
            feed_dict=feed_dict)
        return acc, pred, features, probabilities, tags


if __name__ == "__main__":

    xdata = np.load('data/dzdp_user_feature.npy')
    adj_data = normalize_adj(np.load('data/dzdp_USU_binary_sim.npy'))
    adj_data_1 = normalize_adj(np.load('data/dzdp_UIU_binary_sim.npy'))
    adj_data_2 = normalize_adj(np.load('data/dzdp_URU_binary_sim.npy'))
    # adj_data_3 = normalize_adj(np.load('data/dzdp_UAU_binary_sim.npy'))
    # adj_data_4 = normalize_adj(np.load('data/dzdp_UIU_URU_binary_sim.npy'))
    # adj_data_5 = normalize_adj(np.load('data/dzdp_USU_UIU_binary_sim.npy'))
    # adj_data_6 = normalize_adj(np.load('data/dzdp_USU_URU_binary_sim.npy'))
    # adj_data_7 = normalize_adj(np.load('data/dzdp_USU_UIU_URU_binary_sim.npy'))

    adj_data = np.array([adj_data, adj_data_1, adj_data_2])
    with tf.Session() as sess:

        net = Player2Vec(session=sess, class_size=2, gcn_output1=gcn_para[0],
                         gcn_output2=gcn_para[1], meta=meta_size, nodes=node_size, embedding=node_embedding,
                         encoding=node_encoding)

        sess.run(tf.global_variables_initializer())
        #        net.load(sess)

        t_start = time.clock()
        for epoch in range(epoch_num):
            train_loss = 0
            train_acc = 0
            count = 0
            for index in range(0, train_size, batch_size):
                t = time.clock()
                batch_data, batch_label = get_data(index, batch_size)
                loss, acc, pred, prob = net.train(xdata, adj_data, batch_label,
                                                  batch_data, learning_rate,
                                                  momentum)

                if index % 1 == 0:
                    print("batch loss: {:.4f}, batch acc: {:.4f}".format(loss, acc), "time=",
                          "{:.5f}".format(time.clock() - t))
                train_loss += loss
                train_acc += acc
                count += 1

            train_loss = train_loss / count
            train_acc = train_acc / count
            print("epoch{:d} : train_loss: {:.4f}, train_acc: {:.4f}".format(epoch, train_loss, train_acc))
            # if epoch % 10 == 9:
            #     net.save(sess)

        t_end = time.clock()
        print("train time=", "{:.5f}".format(t_end - t_start))
        print("Train end!")

        test_acc, test_pred, test_features, test_probabilities, test_tags = net.test(xdata, adj_data, test_label,
                                                                                     test_data)

        print("test acc:", test_acc)
