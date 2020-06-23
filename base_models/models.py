import numpy as np
import tensorflow as tf
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
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

