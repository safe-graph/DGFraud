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
from algorithms.Player2Vec.Player2Vec import Player2Vec
from tensorflow.keras import optimizers
from tqdm import tqdm
from utils.data_loader import *
from utils.utils import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# init the common args, expect the model specific args

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123, help='Random seed.')
parser.add_argument('--dataset_str', type=str, default='dblp', help="['dblp','example']")
parser.add_argument('--train_size', type=float, default=0.2, help='training set percentage')
parser.add_argument('--epoch_num', type=int, default=30, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--momentum', type=int, default=0.9)
parser.add_argument('--learning_rate', default=0.001, help='the ratio of training set in whole dataset.')
parser.add_argument('--nhid', type=int, default=128, help='number of hidden units in GCN')
parser.add_argument('--lr', default=0.001, help='learning rate')

args = parser.parse_args()

# set seed
np.random.seed(args.seed)
tf.random.set_seed(args.seed)


def main(support: list, features: tf.SparseTensor, label: tf.Tensor, masks: list, args):
    model = Player2Vec(args.input_dim, args.nhid, args.output_dim, args)
    optimizer = optimizers.Adam(lr=args.lr)

    # train
    for epoch in tqdm(range(args.epoch_num)):
        with tf.GradientTape() as tape:
            train_loss, train_acc = model([support, features, label, masks[0]])

        grads = tape.gradient(train_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # validation
        val_loss, val_acc = model([support, features, label, masks[1]])
        print(f"Epoch: {epoch:d}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f},"
              f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

    # test
    test_loss, test_acc = model([support, features, label, masks[2]])
    print(f"test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}")


if __name__ == "__main__":
    # load the data
    adj_list, features, idx_train, _, idx_val, _, idx_test, _, y = load_data_dblp(meta=True, train_size=args.train_size)
    args.nodes = features.shape[0]

    # convert to dense tensors
    train_mask = tf.convert_to_tensor(sample_mask(idx_train, y.shape[0]))
    val_mask = tf.convert_to_tensor(sample_mask(idx_val, y.shape[0]))
    test_mask = tf.convert_to_tensor(sample_mask(idx_test, y.shape[0]))
    label = tf.convert_to_tensor(y)

    # get sparse tuples
    features = preprocess_feature(features)
    supports = []
    for i in range(len(adj_list)):
        hidden = preprocess_adj(adj_list[i])
        supports.append(hidden)

    # initialize the model parameters
    args.input_dim = features[2][1]
    args.output_dim = y.shape[1]
    args.train_size = len(idx_train)
    args.class_size = y.shape[1]
    args.num_features_nonzero = features[1].shape

    # get sparse tensors
    features = tf.SparseTensor(*features)
    for i in range(len(supports)):
        supports[i] = [tf.cast(tf.SparseTensor(*supports[i]), dtype=tf.float32)]

    main(supports, features, label, [train_mask, val_mask, test_mask], args)
