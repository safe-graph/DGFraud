import numpy as np
import tensorflow as tf
from models.layers import *
flags = tf.app.flags
FLAGS = flags.FLAGS

class Model(object):
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
    def __init__(self, x ,weighted_adj,dim1,dim2,input_dim,output_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = x
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dim1 = dim1
        self.dim2 = dim2
        self.adj = weighted_adj
        self.build()

    def _build(self):
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=self.dim1,
                                            support = self.adj,
                                            act= tf.nn.relu,
                                            dropout=0.0,
                                            sparse_inputs=False,
                                            logging=self.logging,
                                            norm=True))

        self.layers.append(GraphConvolution(input_dim=self.dim1,
                                            output_dim=self.output_dim,
                                            support = self.adj,
                                            act=tf.nn.relu,
                                            dropout=0.,
                                            logging=self.logging,
                                            norm=False))

    def embedding(self):
        return self.outputs
    
class GAT():
    def attention(inputs, attention_size, return_weights=False):
        
        hidden_size = inputs.shape[2].value
    
        # Trainable parameters
        w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    
        with tf.name_scope('v'):
            v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

        vu = tf.tensordot(v, u_omega, axes=1, name='vu')
        weights = tf.nn.softmax(vu, name='alphas')

        output = tf.reduce_sum(inputs * tf.expand_dims(weights, -1), 1)
    
        if not return_weights:
            return output
        else:
            return output, weights