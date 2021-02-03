'''

GEM ('Heterogeneous Graph Neural Networks for Malicious Account Detection')

Parameters:
    input_dim: node feature dim
    output_dim: nodes representation dim
'''
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../..')))

import tensorflow as tf
from tensorflow import keras
from base_models.layers import GEMLayer
from utils import utils
from utils.metrics import *



class GEM(keras.Model):

    def __init__(self, input_dim, output_dim, args):
        super().__init__()

        self.nodes_num = args.nodes_num
        self.class_size = args.class_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hop = args.hop
        self.device_num = args.device_num

    def __call__(self, inputs):

        """:param inputs include support, x, label, mask
        support means a list of the sparse adjacency Tensor
        x means feature
        label means label tensor
        mask means a list of mask tensors to obtain the train data
        """

        support, x, label, idx_mask = inputs

        h_init = tf.keras.initializers.GlorotUniform()
        h = tf.Variable(
                initial_value=h_init(shape=(self.nodes_num, self.output_dim),
                                     dtype='double')
        )

        #forward propagation
        for i in range(0,self.hop):
            f = GEMLayer(self.nodes_num, self.input_dim, self.output_dim, self.device_num)
            gem_out = f((x, support, h))
            h = tf.reshape(gem_out, [x.shape[0], self.output_dim])
        print('GEM embedding over!')

        #classificaiton training process
        masked_data = tf.gather(h, idx_mask)
        masked_label = tf.gather(label, idx_mask)

        #logitstic weights initialize
        W_init = tf.keras.initializers.GlorotUniform()
        u_init = tf.keras.initializers.GlorotUniform()
        b_init = tf.keras.initializers.Zeros()
        W = tf.Variable(
                initial_value=W_init(shape=(self.output_dim, self.class_size),
                                     dtype='double'),
            trainable=True)
        b = tf.Variable(
            initial_value=b_init(shape=(1, self.class_size),
                                 dtype='double'),
            trainable=True)
        u = tf.Variable(
                initial_value=u_init(shape=(1, self.output_dim),
                                     dtype='double'),
            trainable=True)

        """This loss equals to the equation (7) in
	    paper 'Heterogeneous Graph Neural Networks for Malicious Account Detection.'
	    """
        loss = -tf.reduce_sum(tf.math.log(
            tf.nn.sigmoid(masked_label * tf.transpose(tf.matmul(u,tf.transpose(masked_data, perm=[1,0])),
                                                perm=[1,0]))
        ))

        logits = tf.matmul(masked_data, W) + b
        #
        acc = accuracy(logits, masked_label)

        return loss, acc
