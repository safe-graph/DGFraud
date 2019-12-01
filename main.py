# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from algotithm.Player2vec import Player2Vec
import scipy.sparse as sp
import os
import time
import scipy.io as sio

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


tf.reset_default_graph()
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

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


def load_data_dblp(path='data/DBLP4057_GAT_with_idx_tra200_val_800.mat'):
    data = sio.loadmat(path)
    truelabels, features = data['label'], data['features'].astype(float)
    N = features.shape[0]
    rownetworks = [data['net_APA'] - np.eye(N), data['net_APCPA'] - np.eye(N), data['net_APTPA'] - np.eye(N)]
    y = truelabels
    index = range(len(y))
    X_train, X_test, y_train, y_test = train_test_split(index, y, stratify=y, test_size=0.4, random_state=48,
                                                        shuffle=True)

    return rownetworks, features, X_train, y_train, X_test, y_test


adj_list, features, train_data, train_label, test_data, test_label = load_data_dblp()
node_size = features.shape[0]
node_embedding = features.shape[1]
node_encoding = 100
meta_size = len(adj_list)
train_size = len(train_data)


# get batch data
def get_data(ix, int_batch):
    if ix + int_batch >= train_size:
        ix = train_size - int_batch
        end = train_size
    else:
        end = ix + int_batch
    return train_data[ix:end], train_label[ix:end]


# symmetrically normalize adjacency matrix
def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).A


if __name__ == "__main__":
    xdata = features
    adj_data = adj_list

    with tf.Session() as sess:
        net = Player2Vec(session=sess, class_size=4, gcn_output1=gcn_para[0], gcn_output2=gcn_para[1],
                         meta=meta_size, nodes=node_size, embedding=node_embedding, encoding=node_encoding)

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
