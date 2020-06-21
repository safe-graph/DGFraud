'''
This code is due to Yutong Deng (@yutongD), Yingtong Dou (@Yingtong Dou) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection)
https://github.com/safe-graph/DGFraud
'''
import tensorflow as tf
import argparse

from algorithms.GEM.GEM import GEM
from algorithms.GeniePath.GeniePath import GeniePath
from algorithms.Player2Vec.Player2Vec import Player2Vec
from algorithms.FdGars.FdGars import FdGars
from algorithms.SemiGNN.SemiGNN import SemiGNN
from algorithms.GAS.GAS import GAS
import time
from utils.data_loader import *
from utils.utils import *


# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# init the common args, expect the model specific args
def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='GAS',
                        help="['Player2Vec', 'FdGars','GEM','SemiGNN','GAS','GeniePath']")
    parser.add_argument('--seed', type=int, default=123, help='Random seed.')
    parser.add_argument('--dataset_str', type=str, default='example', help="['dblp','example']")
    parser.add_argument('--epoch_num', type=int, default=30, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--momentum', type=int, default=0.9)
    parser.add_argument('--learning_rate', default=0.001, help='the ratio of training set in whole dataset.')

    # GCN args
    parser.add_argument('--hidden1', default=16, help='Number of units in GCN hidden layer 1.')
    parser.add_argument('--hidden2', default=16, help='Number of units in GCN hidden layer 2.')
    parser.add_argument('--gcn_output', default=4, help='gcn output size.')

    # GAS
    parser.add_argument('--review_num sample', default=7, help='review number.')
    parser.add_argument('--gcn_dim', type=int, default=5, help='gcn layer size.')
    parser.add_argument('--encoding1', type=int, default=64)
    parser.add_argument('--encoding2', type=int, default=64)
    parser.add_argument('--encoding3', type=int, default=64)
    parser.add_argument('--encoding4', type=int, default=64)

    # SemiGNN
    parser.add_argument('--init_emb_size', default=4, help='initial node embedding size')
    parser.add_argument('--semi_encoding1', default=3, help='the first view attention layer unit number')
    parser.add_argument('--semi_encoding2', default=2, help='the second view attention layer unit number')
    parser.add_argument('--semi_encoding3', default=4, help='one-layer perceptron units')
    parser.add_argument('--Ul', default=8, help='labeled users number')
    parser.add_argument('--alpha', default=0.5, help='loss alpha')
    parser.add_argument('--lamtha', default=0.5, help='loss lamtha')

    # GEM
    parser.add_argument('--hop', default=1, help='hop number')
    parser.add_argument('--k', default=16, help='gem layer unit')

    # GeniePath
    parser.add_argument('--dim', default=128)
    parser.add_argument('--lstm_hidden', default=128, help='lstm_hidden unit')
    parser.add_argument('--heads', default=1, help='gat heads')
    parser.add_argument('--layer_num', default=4, help='geniePath layer num')

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
        adj_list, features, train_data, train_label, test_data, test_label = load_data_dblp(
            'dataset/DBLP4057_GAT_with_idx_tra200_val_800.mat')
        node_size = features.shape[0]
        node_embedding = features.shape[1]
        class_size = train_label.shape[1]
        train_size = len(train_data)
        paras = [node_size, node_embedding, class_size, train_size]
    if args.dataset_str == 'example' and args.model != 'GAS':
        if args.model == 'GEM':
            adj_list, features, train_data, train_label, test_data, test_label = load_example_gem()
        if args.model == 'SemiGNN':
            adj_list, features, train_data, train_label, test_data, test_label = load_example_semi()
        node_size = features.shape[0]
        node_embedding = features.shape[1]
        class_size = train_label.shape[1]
        train_size = len(train_data)
        paras = [node_size, node_embedding, class_size, train_size]
    if args.dataset_str == 'example' and args.model == 'GAS':
        adj_list, features, train_data, train_label, test_data, test_label = load_data_gas()
        node_embedding_r = features[0].shape[1]
        node_embedding_u = features[1].shape[1]
        node_embedding_i = features[2].shape[1]
        node_size = features[0].shape[0]

        # node_embedding_i = node_embedding_r = node_size
        h_u_size = adj_list[0].shape[1] * (node_embedding_r + node_embedding_u)
        h_i_size = adj_list[2].shape[1] * (node_embedding_r + node_embedding_i)

        class_size = train_label.shape[1]
        train_size = len(train_data)

        paras = [node_size, node_embedding_r, node_embedding_u, node_embedding_i, class_size, train_size, h_u_size,
                 h_i_size]

    return adj_list, features, train_data, train_label, test_data, test_label, paras


def train(args, adj_list, features, train_data, train_label, test_data, test_label, paras):
    with tf.Session() as sess:
        if args.model == 'Player2Vec':
            adj_data = [normalize_adj(adj) for adj in adj_list]
            meta_size = len(adj_list)
            net = Player2Vec(session=sess, class_size=paras[2], gcn_output1=args.hidden1,
                             meta=meta_size, nodes=paras[0], embedding=paras[1], encoding=args.gcn_output)
        if args.model == 'FdGars':
            adj_data = [normalize_adj(adj) for adj in adj_list]
            meta_size = len(adj_list)  # meta=1 in FdGars
            net = FdGars(session=sess, class_size=paras[2], gcn_output1=args.hidden1, gcn_output2=args.hidden2,
                         meta=meta_size, nodes=paras[0], embedding=paras[1], encoding=args.gcn_output)
        if args.model == 'GAS':
            adj_data = adj_list
            net = GAS(session=sess, nodes=paras[0], class_size=paras[4], embedding_r=paras[1], embedding_u=paras[2],
                      embedding_i=paras[3], h_u_size=paras[6], h_i_size=paras[7],
                      encoding1=args.encoding1, encoding2=args.encoding2, encoding3=args.encoding3,
                      encoding4=args.encoding4, gcn_dim=args.gcn_dim)
        if args.model == 'GEM':
            adj_data = adj_list
            meta_size = len(adj_list)  # device num
            net = GEM(session=sess, class_size=paras[2], encoding=args.k,
                      meta=meta_size, nodes=paras[0], embedding=paras[1], hop=args.hop)
        if args.model == 'GeniePath':
            adj_data = adj_list
            net = GeniePath(session=sess, out_dim=paras[2], dim=args.dim, lstm_hidden=args.lstm_hidden,
                            nodes=paras[0], in_dim=paras[1], heads=args.heads, layer_num=args.layer_num,
                            class_size=paras[2])
        if args.model == 'SemiGNN':
            adj_nodelists = [matrix_to_adjlist(adj, pad=False) for adj in adj_list]
            meta_size = len(adj_list)
            pairs = [random_walks(adj_nodelists[i], 2, 3) for i in range(meta_size)]
            net = SemiGNN(session=sess, class_size=paras[2], semi_encoding1=args.semi_encoding1,
                          semi_encoding2=args.semi_encoding2, semi_encoding3=args.semi_encoding3,
                          meta=meta_size, nodes=paras[0], init_emb_size=args.init_emb_size, ul=args.batch_size,
                          alpha=args.alpha, lamtha=args.lamtha)
            adj_data = [pairs_to_matrix(p, paras[0]) for p in pairs]
            u_i = []
            u_j = []
            for adj_nodelist, p in zip(adj_nodelists, pairs):
                u_i_t, u_j_t, graph_label = get_negative_sampling(p, adj_nodelist)
                u_i.append(u_i_t)
                u_j.append(u_j_t)
            u_i = np.concatenate(np.array(u_i))
            u_j = np.concatenate(np.array(u_j))

        sess.run(tf.global_variables_initializer())
        # net.load(sess)

        t_start = time.clock()
        for epoch in range(args.epoch_num):
            train_loss = 0
            train_acc = 0
            count = 0
            for index in range(0, paras[3], args.batch_size):
                if args.model == 'SemiGNN':
                    batch_data, batch_sup_label = get_data(index, args.batch_size, paras[3])
                    loss, acc, pred, prob = net.train(adj_data, u_i, u_j, graph_label, batch_data,
                                                      batch_sup_label,
                                                      args.learning_rate,
                                                      args.momentum)
                else:  # model Player2Vec, GAS， GEM or FdGars
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

        if args.model == 'SemiGNN':
            test_acc, test_pred, test_probabilities, test_tags = net.test(adj_data, u_i, u_j,
                                                                          graph_label,
                                                                          test_data,
                                                                          test_label,
                                                                          args.learning_rate,
                                                                          args.momentum)
        else:
            test_acc, test_pred, test_probabilities, test_tags = net.test(features, adj_data, test_label,
                                                                          test_data)

    print("test acc:", test_acc)


if __name__ == "__main__":
    args = arg_parser()
    set_env(args)
    adj_list, features, train_data, train_label, test_data, test_label, paras = load_data(args)
    train(args, adj_list, features, train_data, train_label, test_data, test_label, paras)
