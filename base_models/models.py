import numpy as np
import tensorflow as tf
from base_models.layers import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    '''Adapted from tkipf/gcn.'''

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None
        self.dim1 = None
        self.dim2 = None
        self.adj = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def embedding(self):
        pass

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class GCN(Model):
    def __init__(self, placeholders, dim1, input_dim, output_dim, index=0, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['x']
        self.placeholders = placeholders
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dim1 = dim1
        self.index = index  # index of meta paths
        self.build()

    def _build(self):
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=self.dim1,
                                            placeholders=self.placeholders,
                                            index=self.index,
                                            act=tf.nn.relu,
                                            dropout=0.0,
                                            sparse_inputs=False,
                                            logging=self.logging,
                                            norm=True))

        self.layers.append(GraphConvolution(input_dim=self.dim1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            index=self.index,
                                            act=tf.nn.relu,
                                            dropout=0.,
                                            logging=self.logging,
                                            norm=False))

    def embedding(self):
        return self.outputs

# class BaseGAttN(object):
#     def loss(logits, labels, nb_classes, class_weights):
#         sample_wts = tf.reduce_sum(tf.multiply(tf.one_hot(labels, nb_classes), class_weights), axis=-1)
#         xentropy = tf.multiply(tf.nn.sparse_softmax_cross_entropy_with_logits(
#             labels=labels, logits=logits), sample_wts)
#         return tf.reduce_mean(xentropy, name='xentropy_mean')
#
#     def training(loss, lr, l2_coef):
#         # weight decay
#         vars = tf.trainable_variables()
#         lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if v.name not
#                            in ['bias', 'gamma', 'b', 'g', 'beta']]) * l2_coef
#
#         # optimizer
#         opt = tf.train.AdamOptimizer(learning_rate=lr)
#
#         # training op
#         train_op = opt.minimize(loss + lossL2)
#
#         return train_op
#
#     def preshape(logits, labels, nb_classes):
#         new_sh_lab = [-1]
#         new_sh_log = [-1, nb_classes]
#         log_resh = tf.reshape(logits, new_sh_log)
#         lab_resh = tf.reshape(labels, new_sh_lab)
#         return log_resh, lab_resh
#
#     def confmat(logits, labels):
#         preds = tf.argmax(logits, axis=1)
#         return tf.confusion_matrix(labels, preds)
#
#
# class GAT(BaseGAttN):
#     def inference(inputs, dim, attn_drop, ffd_drop, bias_mat, n_heads):
#         out = []
#         for i in range(n_heads[-1]):
#             out.append(attn_head(inputs, bias_mat=bias_mat,
#                                  out_sz=dim, activation=tf.nn.elu,
#                                  in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
#         logits = tf.add_n(out) / n_heads[-1]
#         return logits
