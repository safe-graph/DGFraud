'''
This code is due to Yutong Deng (@yutongD), Yingtong Dou (@YingtongDou) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection)
https://github.com/safe-graph/DGFraud
'''
import tensorflow as tf
from tensorflow.keras import optimizers
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../..')))
from algorithms.GAS.GAS import GAS
from utils.data_loader import *
from utils.utils import *

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# init the common args, expect the model specific args
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123, help='Random seed.')
parser.add_argument('--dataset_str', type=str, default='example', help="['dblp','example']")
parser.add_argument('--epoch_num', type=int, default=1, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--momentum', type=int, default=0.9)
parser.add_argument('--lr', default=0.001, help='learning rate')
parser.add_argument('--learning_rate', default=0.001, help='the ratio of training set in whole dataset.')

# GAS
parser.add_argument('--review_num sample', default=7, help='review number.')
parser.add_argument('--gcn_dim', type=int, default=5, help='gcn layer size.')
parser.add_argument('--output_dim1', type=int, default=64)
parser.add_argument('--output_dim2', type=int, default=64)
parser.add_argument('--output_dim3', type=int, default=64)
parser.add_argument('--output_dim4', type=int, default=64)
parser.add_argument('--output_dim5', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')

args = parser.parse_args()

# set seed
np.random.seed(args.seed)
tf.random.set_seed(args.seed)


def main(adj_list: list, r_support: list, features: tf.Tensor, r_feature: tf.SparseTensor, label: tf.Tensor,
         masks: list, args):
    model = GAS(args)
    optimizer = optimizers.Adam(lr=args.lr)

    for epoch in range(args.epoch_num):
        with tf.GradientTape() as tape:
            train_loss, train_acc = model([adj_list, r_support, features, r_feature, label, masks[0]])

        grads = tape.gradient(train_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))


    test_loss, test_acc = model([adj_list, r_support, features, r_feature, label, masks[1]])
    print(f"test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}")


if __name__ == "__main__":
    # load the data
    adj_list, features, X_train, X_test, y = load_data_gas()
    r_support, r_feature = adj_list, features

    r_feature = np.array(features[0], dtype=float)
    r_support = np.array(adj_list[6], dtype=float)

    # convert to dense tensors
    features[0] = tf.convert_to_tensor(features[0], dtype=tf.float32)
    features[1] = tf.convert_to_tensor(features[1], dtype=tf.float32)
    features[2] = tf.convert_to_tensor(features[2], dtype=tf.float32)
    label = tf.convert_to_tensor(y, dtype=tf.float32)

    # get sparse tuples
    r_feature = preprocess_feature(r_feature)
    r_support = preprocess_adj(r_support)

    # initialize the model parameters
    args.reviews_num = features[0].shape[0]
    args.class_size = y.shape[1]
    args.input_dim_i = features[2].shape[1]
    args.input_dim_u = features[1].shape[1]
    args.input_dim_r = features[0].shape[1]
    args.input_dim_r_gcn = r_feature[2][1]
    args.num_features_nonzero = r_feature[1].shape
    args.h_u_size = adj_list[0].shape[1] * (args.input_dim_r + args.input_dim_u)
    args.h_i_size = adj_list[2].shape[1] * (args.input_dim_r + args.input_dim_i)

    # get sparse tensors
    r_feature = tf.SparseTensor(*r_feature)
    r_support = [tf.cast(tf.SparseTensor(*r_support), dtype=tf.float32)]

    masks = [X_train, X_test]

    main(adj_list, r_support, features, r_feature, label, [X_train, X_test], args)
