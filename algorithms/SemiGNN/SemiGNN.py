'''
This code is due to Yutong Deng (@yutongD), Yingtong Dou (@YingtongDou) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection)
https://github.com/safe-graph/DGFraud

SemiGNN ('A Semi-supervised Graph Attentive Network for
        Financial Fraud Detection')

Parameters:
    nodes: total nodes number
    semi_encoding1: the first view attention layer unit number
    semi_encoding2: the second view attention layer unit number
    semi_encoding3: MLP layer unit number
    init_emb_size: the initial node embedding
    meta: view number
    ul: labeled users number
'''
import os
import sys
from utils.metrics import accuracy
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../..')))
import tensorflow as tf
from tensorflow import keras
from base_models.layers import AttentionLayer


class SemiGNN(keras.Model):
    def __init__(self, args):
        super().__init__()
        self.nodes = args.nodes
        self.class_size = args.class_size
        self.semi_encoding1 = args.semi_encoding1
        self.semi_encoding2 = args.semi_encoding2
        self.semi_encoding3 = args.semi_encoding3
        self.init_emb_size = args.init_emb_size
        self.meta = args.meta
        self.batch_size = args.batch_size
        self.alpha = args.alpha
        self.lamtha = args.lamtha

        # init embedding
        self.x_init = tf.keras.initializers.GlorotUniform()
        self.u = tf.Variable(initial_value=self.x_init(shape=(self.semi_encoding3, self.class_size), dtype=tf.float32),
                            trainable=True)

    def __call__(self, inputs):
        adj_data, u_i, u_j, graph_label, label, idx_mask = inputs

        h1 = []
        for i in range(self.meta):
            emb = tf.Variable(initial_value=self.x_init(shape=(self.nodes, self.init_emb_size), dtype=tf.float32),
                              trainable=True)
            h = AttentionLayer.node_attention(inputs=emb, adj=adj_data[i])
            h = tf.reshape(h, [self.nodes, emb.shape[1]])
            h1.append(h)
        h1 = tf.concat(h1, 0)
        h1 = tf.reshape(h1, [self.meta, self.nodes, self.init_emb_size])

        h2 = AttentionLayer.view_attention(inputs=h1, layer_size=2,
                                           meta=self.meta, encoding1=self.semi_encoding1,
                                           encoding2=self.semi_encoding2)
        h2 = tf.reshape(h2, [self.nodes, self.semi_encoding2 * self.meta])

        a_u = tf.keras.layers.Dense(self.semi_encoding3)(h2)

        masked_data = tf.gather(a_u, idx_mask)
        masked_label = tf.gather(label, idx_mask)
        logits = tf.nn.softmax(tf.matmul(masked_data, self.u))

        loss1 = -(1 / self.batch_size) * tf.reduce_sum(masked_label * tf.math.log(tf.nn.softmax(logits)))
        u_i_embedding = tf.nn.embedding_lookup(a_u, tf.cast(u_i, dtype=tf.int32))
        u_j_embedding = tf.nn.embedding_lookup(a_u, tf.cast(u_j, dtype=tf.int32))
        inner_product = tf.reduce_sum(u_i_embedding * u_j_embedding, axis=1)
        loss2 = -tf.reduce_mean(tf.math.log_sigmoid(graph_label * inner_product))
        loss = self.alpha * loss1 + (1 - self.alpha) * loss2
        acc = accuracy(logits, masked_label)

        return loss, acc
