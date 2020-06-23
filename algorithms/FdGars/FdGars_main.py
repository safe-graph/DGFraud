'''
This code is due to Yutong Deng (@yutongD), Yingtong Dou (@YingtongDou) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection)
https://github.com/safe-graph/DGFraud
'''

import tensorflow as tf
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../..')))
from algorithms.FdGars.FdGars import FdGars
import time
from utils.data_loader import *
from utils.utils import *


# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# init the common args, expect the model specific args
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123, help='Random seed.')
    parser.add_argument('--dataset_str', type=str, default='dblp', help="['dblp','example']")
    parser.add_argument('--epoch_num', type=int, default=30, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--momentum', type=int, default=0.9)
    parser.add_argument('--learning_rate', default=0.001, help='the ratio of training set in whole dataset.')

    # GCN args
    parser.add_argument('--hidden1', default=16, help='Number of units in GCN hidden layer 1.')
    parser.add_argument('--hidden2', default=16, help='Number of units in GCN hidden layer 2.')
    parser.add_argument('--gcn_output', default=4, help='gcn output size.')

    args = parser.parse_args()
    return args


def set_env(args):
    tf.reset_default_graph()
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)


# get batch data
def get_data(ix, int_batch, train_size):
    if ix + int_batch >= train_size:
        ix = train_size - int_batch
        end = train_size
    else:
        end = ix + int_batch
    return train_data[ix:end], train_label[ix:end]


def load_data(args):
    if args.dataset_str == 'dblp':
        adj_list, features, train_data, train_label, test_data, test_label = load_data_dblp()
        node_size = features.shape[0]
        node_embedding = features.shape[1]
        class_size = train_label.shape[1]
        train_size = len(train_data)
        paras = [node_size, node_embedding, class_size, train_size]

    return adj_list, features, train_data, train_label, test_data, test_label, paras


def train(args, adj_list, features, train_data, train_label, test_data, test_label, paras):
    with tf.Session() as sess:
        adj_data = [normalize_adj(adj) for adj in adj_list]
        meta_size = len(adj_list)  # meta=1 in FdGars
        net = FdGars(session=sess, class_size=paras[2], gcn_output1=args.hidden1, gcn_output2=args.hidden2,
                     meta=meta_size, nodes=paras[0], embedding=paras[1], encoding=args.gcn_output)

        sess.run(tf.global_variables_initializer())
        # net.load(sess)

        t_start = time.clock()
        for epoch in range(args.epoch_num):
            train_loss = 0
            train_acc = 0
            count = 0
            for index in range(0, paras[3], args.batch_size):
                batch_data, batch_label = get_data(index, args.batch_size, paras[3])
                loss, acc, pred, prob = net.train(features, adj_data, batch_label,
                                                  batch_data, args.learning_rate,
                                                  args.momentum)

                print("batch loss: {:.4f}, batch acc: {:.4f}".format(loss, acc))
                # print(prob, pred)

                train_loss += loss
                train_acc += acc
                count += 1
            train_loss = train_loss / count
            train_acc = train_acc / count
            print("epoch{:d} : train_loss: {:.4f}, train_acc: {:.4f}".format(epoch, train_loss, train_acc))

            # net.save(sess)

        t_end = time.clock()
        print("train time=", "{:.5f}".format(t_end - t_start))
        print("Train end!")

        test_acc, test_pred, test_probabilities, test_tags = net.test(features, adj_data, test_label,
                                                                      test_data)

    print("test acc:", test_acc)


if __name__ == "__main__":
    args = arg_parser()
    set_env(args)
    adj_list, features, train_data, train_label, test_data, test_label, paras = load_data(args)
    train(args, adj_list, features, train_data, train_label, test_data, test_label, paras)
