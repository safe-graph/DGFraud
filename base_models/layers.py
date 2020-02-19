from base_models.inits import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


##########################
# Adapted from tkipf/gcn #
##########################

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

    def __init__(self, input_dim, output_dim, support, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, norm=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        self.dropout = dropout
        self.act = act
        self.support = support
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.norm = norm

        # helper variable for sparse dropout
        self.num_features_nonzero = np.ones(4637, dtype='int32')

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
            support = dot(self.support, pre_sup, sparse=False)
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


class ScaledDotProductAttentionLayer(Layer):
    def attention(q, k, v, mask):
        qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention = qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention += 1

        weights = tf.nn.softmax(scaled_attention, axis=-1)
        output = tf.matmul(weights, v)

        return output, weights


class SimpleAttLayer(Layer):
    def attention(inputs, attention_size, return_weights=False):
        inputs = tf.expand_dims(inputs, 0)
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


class ConcatenationAggregator(Layer):
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
                                    tf.cast(self.review_item_adj, dtype=tf.int32))  # input is int, why need cast?
        ri = tf.transpose(tf.random_shuffle(tf.transpose(ri)))

        ru = tf.nn.embedding_lookup(user_vecs, tf.cast(self.review_user_adj, dtype=tf.int32))
        ru = tf.transpose(tf.random_shuffle(tf.transpose(ru)))

        concate_vecs = tf.concat([review_vecs, ru, ri], axis=1)

        # [nodes] x [out_dim]
        output = tf.matmul(concate_vecs, self.vars['con_agg_weights'])

        return self.act(output)


class AttentionAggregator(Layer):

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
        concate_user_vecs, _ = ScaledDotProductAttentionLayer.attention(q=user_vecs, k=user_vecs, v=concate_user_vecs,
                                                                        mask=None)
        concate_item_vecs, _ = ScaledDotProductAttentionLayer.attention(q=item_vecs, k=item_vecs, v=concate_item_vecs,
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
