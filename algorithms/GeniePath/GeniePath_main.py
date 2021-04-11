import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../..')))

import tensorflow as tf
from tensorflow.keras import optimizers

import argparse
import time
from tqdm import tqdm

from algorithms.GeniePath.GeniePath import GeniePath

from utils.data_loader import *
from utils.utils import *


# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# init the common args, expect the model specific args
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123, help='Random seed.')
parser.add_argument('--dataset_str', type=str, default='dblp', help="['dblp','example']")
parser.add_argument('--epoch_num', type=int, default=30, help='Number of epochs to train.')
parser.add_argument('--train_size', type=float, default=0.2, help='training set percentage')
parser.add_argument('--momentum', type=int, default=0.9)
parser.add_argument('--lr', default=0.001, help='learning rate')

# GeniePath
parser.add_argument('--GAT_output_dim', default=128)
parser.add_argument('--lstm_hidden', default=128, help='lstm_hidden unit')
parser.add_argument('--GAT_heads', default=1, help='gat heads')
parser.add_argument('--layer_num', default=4, help='geniePath layer num')

args = parser.parse_args()

# set seed
np.random.seed(args.seed)
tf.random.set_seed(args.seed)


def main(supports: list, features: tf.SparseTensor, label: tf.Tensor, masks: list, args):
    """
    @param supports: a list of the sparse adjacency matrix
    @param features: the feature of the sparse tensor for all nodes
    @param label: the label tensor for all nodes
    @param masks: a list of mask tensors to obtain the train, val, and test data
    @param args: additional parameters
    """
    model = GeniePath(args.input_dim, args.output_dim, args)
    optimizer = optimizers.Adam(lr=args.lr)

    # train
    for epoch in tqdm(range(args.epoch_num)):

        with tf.GradientTape() as tape:
            train_loss, train_acc = model([supports, features, label, masks[0]])

        grads = tape.gradient(train_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # validation
        val_loss, val_acc = model([supports, features, label, masks[1]])
        print(f"Epoch: {epoch:d}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f},"
            f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

    # test
    test_loss, test_acc = model([supports, features, label, masks[2]])
    print(f"test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}")


if __name__ == "__main__":
    adj_list, features, idx_train, _, idx_val, _, idx_test, _, y = load_data_dblp(meta=False, train_size=args.train_size)

    # convert to dense tensors
    label = tf.convert_to_tensor(y, dtype=tf.float32)
    features = tf.convert_to_tensor(features, dtype=tf.float32)
    supports = tf.convert_to_tensor(adj_list, dtype=tf.float32)

    # initialize the model parameters
    args.input_dim = features.shape[1]
    args.nodes_num = features.shape[0]
    args.output_dim = y.shape[1]
    args.class_size = y.shape[1]
    args.train_size = len(idx_train)
    args.device_num = len(adj_list)

    # features = preprocess_feature(features)
    # supports = [preprocess_adj(adj) for adj in adj_list]

    # get sparse tensors
    #features = tf.cast(tf.Tensor(*features), dtype=tf.float32)
    #supports = [tf.cast(tf.Tensor(*support), dtype=tf.float32) for support in supports]

    masks = [idx_train, idx_val, idx_test]

    main(supports, features, label, masks, args)
