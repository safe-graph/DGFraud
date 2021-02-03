import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../..')))

import tensorflow as tf
from tensorflow.keras import optimizers

import argparse
import time
from tqdm import tqdm

from algorithms.GEM.GEM import GEM

from utils.data_loader import *
from utils.utils import *



# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# init the common args, expect the model specific args
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123, help='Random seed.')
parser.add_argument('--dataset_str', type=str, default='dblp', help="['dblp','example']")
parser.add_argument('--train_size', type=float, default=0.8, help='training set percentage')
parser.add_argument('--epoch_num', type=int, default=30, help='Number of epochs to train.')
parser.add_argument('--momentum', type=int, default=0.9)
parser.add_argument('--lr', default=0.001, help='learning rate')

# GEM
parser.add_argument('--hop', default=1, help='hop number')
parser.add_argument('--output_dim', default=16, help='gem layer unit')

args = parser.parse_args()

#set seed
np.random.seed(args.seed)
tf.random.set_seed(args.seed)

def main(support: list, features: tf.SparseTensor, label: tf.Tensor, masks: list, args):
    """
    @param support: a list of the sparse adjacency matrix
    @param features: the feature of the sparse tensor for all nodes
    @param label: the label tensor for all nodes
    @param masks: a list of mask tensors to obtain the train, val, and test data
    @param args: additional parameters
    """
    model = GEM(args.input_dim, args.output_dim, args)
    optimizer = optimizers.Adam(lr=args.lr)

    #train
    for epoch in tqdm(range(args.epoch_num)):

        with tf.GradientTape() as tape:
            train_loss, train_acc = model([support, features, label, masks[0]])

        grads = tape.gradient(train_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        #validation
        val_loss, val_acc = model([support, features, label, masks[1]])
        print(f"Epoch: {epoch:d}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f},"
              f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

    #test
    test_loss, test_acc = model([support, features, label, masks[2]])
    print(f"test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}")





if __name__ == "__main__":
    adj_list, features, idx_train, _, idx_val, _, idx_test, _, y = load_data_dblp(meta=True, train_size=args.train_size)

    # convert to dense tensors
    # train_mask = tf.convert_to_tensor(sample_mask(idx_train, y.shape[0]))
    # val_mask = tf.convert_to_tensor(sample_mask(idx_val, y.shape[0]))
    # test_mask = tf.convert_to_tensor(sample_mask(idx_test, y.shape[0]))
    label = tf.convert_to_tensor(y)

    #initialize the model parameters
    args.input_dim = features.shape[1]
    args.nodes_num = features.shape[0]
    args.class_size = y.shape[1]
    args.train_size = len(idx_train)
    args.device_num = len(adj_list)

    #use the whole graph
    idx_train = np.arange(args.nodes_num)
    idx_val = np.arange(args.nodes_num)
    idx_test = np.arange(args.nodes_num)

    main(adj_list, features, label, [idx_train, idx_val, idx_test], args)

