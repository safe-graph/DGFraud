'''
This code is due to Yutong Deng (@yutongD), Yingtong Dou (@YingtongDou) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection)
https://github.com/safe-graph/DGFraud
'''
from base_models.inits import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}

'''Code about GCN is adapted from tkipf/gcn.'''


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def _call(self, inputs, adj_info):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class GraphConvolution(Layer):
    """Graph convolution layer."""

    def __init__(self, input_dim, output_dim, placeholders, index=0, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, norm=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        self.dropout = dropout
        self.act = act
        self.support = placeholders['a']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.norm = norm
        self.index = index

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(1):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        # convolve
        supports = list()
        for i in range(1):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[self.index], pre_sup, sparse=False)
            supports.append(support)
        output = tf.add_n(supports)
        axis = list(range(len(output.get_shape()) - 1))
        mean, variance = tf.nn.moments(output, axis)
        scale = None
        offset = None
        variance_epsilon = 0.001
        output = tf.nn.batch_normalization(output, mean, variance, offset, scale, variance_epsilon)

        # bias
        if self.bias:
            output += self.vars['bias']
        if self.norm:
            # return self.act(output)/tf.reduce_sum(self.act(output))
            return tf.nn.l2_normalize(self.act(output), axis=None, epsilon=1e-12)
        return self.act(output)


class AttentionLayer(Layer):
    """ AttentionLayer is a function f : hkey × Hval → hval which maps
    a feature vector hkey and the set of candidates’ feature vectors
    Hval to an weighted sum of elements in Hval.
    """

    def attention(inputs, attention_size, v_type=None, return_weights=False, bias=True, joint_type='weighted_sum',
                  multi_view=True):
        if multi_view:
            inputs = tf.expand_dims(inputs, 0)
        hidden_size = inputs.shape[-1].value

        # Trainable parameters
        w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

        with tf.name_scope('v'):
            v = tf.tensordot(inputs, w_omega, axes=1)
            if bias is True:
                v += b_omega
            if v_type is 'tanh':
                v = tf.tanh(v)
            if v_type is 'relu':
                v = tf.nn.relu(v)

        vu = tf.tensordot(v, u_omega, axes=1, name='vu')
        weights = tf.nn.softmax(vu, name='alphas')

        if joint_type is 'weighted_sum':
            output = tf.reduce_sum(inputs * tf.expand_dims(weights, -1), 1)
        if joint_type is 'concatenation':
            output = tf.concat(inputs * tf.expand_dims(weights, -1), 2)

        if not return_weights:
            return output
        else:
            return output, weights

    def node_attention(inputs, adj, return_weights=False):
        hidden_size = inputs.shape[-1].value
        H_v = tf.Variable(tf.random_normal([hidden_size, 1], stddev=0.1))

        # convert adj to sparse tensor
        zero = tf.constant(0, dtype=tf.float32)
        where = tf.not_equal(adj, zero)
        indices = tf.where(where)
        values = tf.gather_nd(adj, indices)
        adj = tf.SparseTensor(indices=indices,
                              values=values,
                              dense_shape=adj.shape)

        with tf.name_scope('v'):
            v = adj * tf.squeeze(tf.tensordot(inputs, H_v, axes=1))

        weights = tf.sparse_softmax(v, name='alphas')  # [nodes,nodes]
        output = tf.sparse_tensor_dense_matmul(weights, inputs)

        if not return_weights:
            return output
        else:
            return output, weights

    # view-level attention (equation (4) in SemiGNN)
    def view_attention(inputs, encoding1, encoding2, layer_size, meta, return_weights=False):
        h = inputs
        encoding = [encoding1, encoding2]
        for l in range(layer_size):
            v = []
            for i in range(meta):
                input = h[i]
                v_i = tf.layers.dense(inputs=input, units=encoding[l], activation=tf.nn.relu)
                v.append(v_i)
            h = v

        h = tf.concat(h, 0)
        h = tf.reshape(h, [meta, inputs[0].shape[0].value, encoding2])
        phi = tf.Variable(tf.random_normal([encoding2, ], stddev=0.1))
        weights = tf.nn.softmax(h * phi, name='alphas')
        output = tf.reshape(h * weights, [1, inputs[0].shape[0] * encoding2 * meta])

        if not return_weights:
            return output
        else:
            return output, weights

    def scaled_dot_product_attention(q, k, v, mask):
        qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention = qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention += 1

        weights = tf.nn.softmax(scaled_attention, axis=-1)
        output = tf.matmul(weights, v)
        return output, weights


class ConcatenationAggregator(Layer):
    """This layer equals to the equation (3) in
    paper 'Spam Review Detection with Graph Convolutional Networks.'
    """

    def __init__(self, input_dim, output_dim, review_item_adj, review_user_adj,
                 review_vecs, user_vecs, item_vecs, dropout=0., act=tf.nn.relu,
                 name=None, concat=False, **kwargs):
        super(ConcatenationAggregator, self).__init__(**kwargs)

        self.review_item_adj = review_item_adj
        self.review_user_adj = review_user_adj
        self.review_vecs = review_vecs
        self.user_vecs = user_vecs
        self.item_vecs = item_vecs

        self.dropout = dropout
        self.act = act
        self.concat = concat

        if name is not None:
            name = '/' + name
        else:
            name = ''

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['con_agg_weights'] = glorot([input_dim, output_dim],
                                                  name='con_agg_weights')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs):
        review_vecs = tf.nn.dropout(self.review_vecs, 1 - self.dropout)
        user_vecs = tf.nn.dropout(self.user_vecs, 1 - self.dropout)
        item_vecs = tf.nn.dropout(self.item_vecs, 1 - self.dropout)

        # neighbor sample
        ri = tf.nn.embedding_lookup(item_vecs,
                                    tf.cast(self.review_item_adj, dtype=tf.int32))
        ri = tf.transpose(tf.random_shuffle(tf.transpose(ri)))

        ru = tf.nn.embedding_lookup(user_vecs, tf.cast(self.review_user_adj, dtype=tf.int32))
        ru = tf.transpose(tf.random_shuffle(tf.transpose(ru)))

        concate_vecs = tf.concat([review_vecs, ru, ri], axis=1)

        # [nodes] x [out_dim]
        output = tf.matmul(concate_vecs, self.vars['con_agg_weights'])

        return self.act(output)


class AttentionAggregator(Layer):
    """This layer equals to equation (5) and equation (8) in
    paper 'Spam Review Detection with Graph Convolutional Networks.'
    """

    def __init__(self, input_dim1, input_dim2, output_dim, hid_dim, user_review_adj, user_item_adj, item_review_adj,
                 item_user_adj,
                 review_vecs, user_vecs, item_vecs, dropout=0., bias=False, act=tf.nn.relu,
                 name=None, concat=False, **kwargs):
        super(AttentionAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat
        self.user_review_adj = user_review_adj
        self.user_item_adj = user_item_adj
        self.item_review_adj = item_review_adj
        self.item_user_adj = item_user_adj
        self.review_vecs = review_vecs
        self.user_vecs = user_vecs
        self.item_vecs = item_vecs
        if name is not None:
            name = '/' + name
        else:
            name = ''

        with tf.variable_scope(self.name + name + '_vars'):

            self.vars['user_weights'] = glorot([input_dim1, hid_dim],
                                               name='user_weights')
            self.vars['item_weights'] = glorot([input_dim2, hid_dim],
                                               name='item_weights')
            self.vars['concate_user_weights'] = glorot([hid_dim, output_dim],
                                                       name='user_weights')
            self.vars['concate_item_weights'] = glorot([hid_dim, output_dim],
                                                       name='item_weights')

            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.output_dim = output_dim

    def _call(self, inputs):

        review_vecs = tf.nn.dropout(self.review_vecs, 1 - self.dropout)
        user_vecs = tf.nn.dropout(self.user_vecs, 1 - self.dropout)
        item_vecs = tf.nn.dropout(self.item_vecs, 1 - self.dropout)

        # num_samples = self.adj_info[4]

        # neighbor sample
        ur = tf.nn.embedding_lookup(review_vecs, tf.cast(self.user_review_adj, dtype=tf.int32))
        ur = tf.transpose(tf.random_shuffle(tf.transpose(ur)))
        # ur = tf.slice(ur, [0, 0], [-1, num_samples])

        ri = tf.nn.embedding_lookup(item_vecs, tf.cast(self.user_item_adj, dtype=tf.int32))
        ri = tf.transpose(tf.random_shuffle(tf.transpose(ri)))
        # ri = tf.slice(ri, [0, 0], [-1, num_samples])

        ir = tf.nn.embedding_lookup(review_vecs, tf.cast(self.item_review_adj, dtype=tf.int32))
        ir = tf.transpose(tf.random_shuffle(tf.transpose(ir)))
        # ir = tf.slice(ir, [0, 0], [-1, num_samples])

        ru = tf.nn.embedding_lookup(user_vecs, tf.cast(self.item_user_adj, dtype=tf.int32))
        ru = tf.transpose(tf.random_shuffle(tf.transpose(ru)))
        # ru = tf.slice(ru, [0, 0], [-1, num_samples])

        concate_user_vecs = tf.concat([ur, ri], axis=2)
        concate_item_vecs = tf.concat([ir, ru], axis=2)

        # concate neighbor's embedding
        s1 = tf.shape(concate_user_vecs)
        s2 = tf.shape(concate_item_vecs)
        concate_user_vecs = tf.reshape(concate_user_vecs, [s1[0], s1[1] * s1[2]])
        concate_item_vecs = tf.reshape(concate_item_vecs, [s2[0], s2[1] * s2[2]])

        # attention
        concate_user_vecs, _ = AttentionLayer.scaled_dot_product_attention(q=user_vecs, k=user_vecs,
                                                                           v=concate_user_vecs,
                                                                           mask=None)
        concate_item_vecs, _ = AttentionLayer.scaled_dot_product_attention(q=item_vecs, k=item_vecs,
                                                                           v=concate_item_vecs,
                                                                           mask=None)

        # [nodes] x [out_dim]
        user_output = tf.matmul(concate_user_vecs, self.vars['user_weights'])
        item_output = tf.matmul(concate_item_vecs, self.vars['item_weights'])

        # bias
        if self.bias:
            user_output += self.vars['bias']
            item_output += self.vars['bias']

        user_output = self.act(user_output)
        item_output = self.act(item_output)

        #  Combination
        if self.concat:
            user_output = tf.matmul(user_output, self.vars['concate_user_weights'])
            item_output = tf.matmul(item_output, self.vars['concate_item_weights'])

            user_output = tf.concat([user_vecs, user_output], axis=1)
            item_output = tf.concat([item_vecs, item_output], axis=1)

        return user_output, item_output


class GASConcatenation(Layer):
    """GCN-based Anti-Spam(GAS) layer for concatenation of comment embedding learned by GCN from the Comment Graph
     and other embeddings learned in previous operations.
     """

    def __init__(self, review_item_adj, review_user_adj,
                 review_vecs, item_vecs, user_vecs, homo_vecs, name=None, **kwargs):
        super(GASConcatenation, self).__init__(**kwargs)

        self.review_item_adj = review_item_adj
        self.review_user_adj = review_user_adj
        self.review_vecs = review_vecs
        self.user_vecs = user_vecs
        self.item_vecs = item_vecs
        self.homo_vecs = homo_vecs

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        # neighbor sample
        ri = tf.nn.embedding_lookup(self.item_vecs, tf.cast(self.review_item_adj, dtype=tf.int32))
        # ri = tf.transpose(tf.random_shuffle(tf.transpose(ri)))
        # ir = tf.slice(ir, [0, 0], [-1, num_samples])

        ru = tf.nn.embedding_lookup(self.user_vecs, tf.cast(self.review_user_adj, dtype=tf.int32))
        # ru = tf.transpose(tf.random_shuffle(tf.transpose(ru)))
        # ru = tf.slice(ru, [0, 0], [-1, num_samples])

        concate_vecs = tf.concat([ri, self.review_vecs, ru, self.homo_vecs], axis=1)
        return concate_vecs


class GEMLayer(Layer):
    """This layer equals to the equation (8) in
    paper 'Heterogeneous Graph Neural Networks for Malicious Account Detection.'
    """

    def __init__(self, placeholders, nodes, device_num, embedding, encoding, name=None, **kwargs):
        super(GEMLayer, self).__init__(**kwargs)

        self.nodes = nodes
        self.devices_num = device_num
        self.encoding = encoding
        self.embedding = embedding
        self.placeholders = placeholders

        if name is not None:
            name = '/' + name
        else:
            name = ''

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['W'] = glorot([embedding, encoding], name='W')
            self.vars['V'] = glorot([encoding, encoding], name='V')
            self.vars['alpha'] = glorot([self.devices_num, 1], name='V')
        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        h1 = tf.matmul(self.placeholders['x'], self.vars['W'])
        h2 = []
        for d in range(self.devices_num):
            ahv = tf.matmul(tf.matmul(self.placeholders['a'][d], inputs), self.vars['V'])
            h2.append(ahv)
        h2 = tf.concat(h2, 0)
        h2 = tf.reshape(h2, [self.devices_num, self.nodes * self.encoding])
        h2 = tf.transpose(h2, [1, 0])
        h2 = tf.reshape(tf.matmul(h2, tf.nn.softmax(self.vars['alpha'])), [self.nodes, self.encoding])

        h = tf.nn.sigmoid(h1 + h2)
        return h


class GAT(Layer):
    """This layer is adapted from PetarV-/GAT.'
    """

    def __init__(self, dim, attn_drop, ffd_drop, bias_mat, n_heads, name=None, **kwargs):
        super(GAT, self).__init__(**kwargs)

        self.dim = dim
        self.attn_drop = attn_drop
        self.ffd_drop = ffd_drop
        self.bias_mat = bias_mat
        self.n_heads = n_heads

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if self.logging:
            self._log_vars()

    def attn_head(self, seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False):
        conv1d = tf.layers.conv1d
        with tf.name_scope('my_attn'):
            if in_drop != 0.0:
                seq = tf.nn.dropout(seq, 1.0 - in_drop)

            seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

            # simplest self-attention possible
            f_1 = tf.layers.conv1d(seq_fts, 1, 1)
            f_2 = tf.layers.conv1d(seq_fts, 1, 1)
            logits = f_1 + tf.transpose(f_2, [0, 2, 1])
            coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

            if coef_drop != 0.0:
                coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
            if in_drop != 0.0:
                seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

            vals = tf.matmul(coefs, seq_fts)
            ret = tf.contrib.layers.bias_add(vals)

            # residual connection
            if residual:
                if seq.shape[-1] != ret.shape[-1]:
                    ret = ret + conv1d(seq, ret.shape[-1], 1)
                else:
                    ret = ret + seq

            return activation(ret)

    def inference(self, inputs):
        out = []
        # for i in range(n_heads[-1]):
        for i in range(self.n_heads):
            out.append(self.attn_head(inputs, bias_mat=self.bias_mat, out_sz=self.dim, activation=tf.nn.elu,
                                      in_drop=self.ffd_drop, coef_drop=self.attn_drop, residual=False))
        logits = tf.add_n(out) / self.n_heads
        return logits


class GeniePathLayer(Layer):
    """This layer equals to the Adaptive Path Layer in
    paper 'GeniePath: Graph Neural Networks with Adaptive Receptive Paths.'
    The code is adapted from shawnwang-tech/GeniePath-pytorch
    """

    def __init__(self, placeholders, nodes, in_dim, dim, heads=1, name=None, **kwargs):
        super(GeniePathLayer, self).__init__(**kwargs)

        self.nodes = nodes
        self.in_dim = in_dim
        self.dim = dim
        self.heads = heads
        self.placeholders = placeholders

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if self.logging:
            self._log_vars()

    def depth_forward(self, x, h, c):
        with tf.variable_scope('lstm', reuse=tf.AUTO_REUSE):
            cell = tf.nn.rnn_cell.LSTMCell(num_units=h, state_is_tuple=True)
            x, (c, h) = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
        return x, (c, h)

    def breadth_forward(self, x, bias_in):
        x = tf.tanh(GAT(self.dim, attn_drop=0, ffd_drop=0, bias_mat=bias_in, n_heads=self.heads).inference(x))
        return x

    def forward(self, x, bias_in, h, c):
        x = self.breadth_forward(x, bias_in)
        x, (h, c) = self.depth_forward(x, h, c)
        x = x[0]
        return x, (h, c)

    # def lazy_forward(self, x, bias_in, h, c):
    #     x = self.breadth_forward(x, bias_in)
    #     x, (h, c) = self.depth_forward(x, h, c)
    #     x = x[0]
    #     return x, (h, c)
