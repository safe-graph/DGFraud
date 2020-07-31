'''
This code is due to Zhiwei Liu (@JimLiu96) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection)
https://github.com/safe-graph/DGFraud
'''
import tensorflow as tf
import models as models
import layers as layers
from aggregators import MeanAggregator, MaxPoolingAggregator, MeanPoolingAggregator, SeqAggregator, GCNAggregator
from inits import glorot, zeros

flags = tf.app.flags
FLAGS = flags.FLAGS


class SupervisedGraphconsis(models.SampleAndAggregate):
    """Implementation of supervised GraphConsis."""

    def __init__(self, num_classes,
            placeholders, features, adj, degrees,
            layer_infos, concat=True, aggregator_type="mean", 
            model_size="small", sigmoid_loss=False, identity_dim=0, num_re=3,
                **kwargs):
        '''
        Args:
            - placeholders: Stanford TensorFlow placeholder object.
            - features: Numpy array with node features.
            - adj: Numpy array with adjacency lists (padded with random re-samples)
            - degrees: Numpy array with node degrees. 
            - layer_infos: List of SAGEInfo namedtuples that describe the parameters of all 
                   the recursive layers. See SAGEInfo definition above. It contains *numer_re* lists of layer_info
            - concat: whether to concatenate during recursive iterations
            - aggregator_type: how to aggregate neighbor information
            - model_size: one of "small" and "big"
            - sigmoid_loss: Set to true if nodes can belong to multiple classes
            - identity_dim: context embedding
        '''

        models.GeneralizedModel.__init__(self, **kwargs)

        if aggregator_type == "mean":
            self.aggregator_cls = MeanAggregator
        elif aggregator_type == "seq":
            self.aggregator_cls = SeqAggregator
        elif aggregator_type == "meanpool":
            self.aggregator_cls = MeanPoolingAggregator
        elif aggregator_type == "maxpool":
            self.aggregator_cls = MaxPoolingAggregator
        elif aggregator_type == "gcn":
            self.aggregator_cls = GCNAggregator
        else:
            raise Exception("Unknown aggregator: ", self.aggregator_cls)

        # get info from placeholders...
        self.inputs1 = placeholders["batch"]
        self.model_size = model_size
        self.adj_info = adj
        if identity_dim > 0:
           self.embeds_context = tf.get_variable("node_embeddings", [features.shape[0], identity_dim])
        else:
           self.embeds_context = None
        if features is None: 
            if identity_dim == 0:
                raise Exception("Must have a positive value for identity feature dimension if no input features given.")
            self.features = self.embeds_context
        else:
            self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
            if not self.embeds_context is None:
                self.features = tf.concat([self.embeds_context, self.features], axis=1)
        self.degrees = degrees
        self.concat = concat
        self.num_classes = num_classes
        self.sigmoid_loss = sigmoid_loss
        self.dims = [(0 if features is None else features.shape[1]) + identity_dim]
        self.dims.extend([layer_infos[0][i].output_dim for i in range(len(layer_infos[0]))])
        self.batch_size = placeholders["batch_size"]
        self.placeholders = placeholders
        self.layer_infos = layer_infos
        self.num_relations = num_re
        dim_mult = 2 if self.concat else 1
        self.relation_vectors = tf.Variable(glorot([num_re, self.dims[-1] * dim_mult]), trainable=True, name='relation_vectors')
        self.attention_vec = tf.Variable(glorot([self.dims[-1] * dim_mult * 2, 1]))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.build()


    def build(self):
        samples1_list, support_sizes1_list = [], []
        for r_idx in range(self.num_relations):
            samples1, support_sizes1 = self.sample(self.inputs1, self.layer_infos[r_idx])
            samples1_list.append(samples1)
            support_sizes1_list.append(support_sizes1)
        num_samples = [layer_info.num_samples for layer_info in self.layer_infos[0]]
        self.outputs1_list = []
        dim_mult = 2 if self.concat else 1 # multiplication to get the correct output dimension
        dim_mult = dim_mult * 2
        for r_idx in range(self.num_relations):
            outputs1, self.aggregators = self.aggregate(samples1_list[r_idx], [self.features], self.dims, num_samples,
                support_sizes1, concat=self.concat, model_size=self.model_size)
            self.relation_batch = tf.tile([tf.nn.embedding_lookup(self.relation_vectors, r_idx)], [self.batch_size, 1])
            outputs1 = tf.concat([outputs1, self.relation_batch], 1)
            self.attention_weights = tf.matmul(outputs1, self.attention_vec)
            self.attention_weights = tf.tile(self.attention_weights, [1, dim_mult*self.dims[-1]])
            outputs1 = tf.multiply(self.attention_weights, outputs1)
            self.outputs1_list += [outputs1]
        # self.outputs1 = tf.reduce_mean(self.outputs1_list, 0)
        self.outputs1 = tf.stack(self.outputs1_list, 1)
        self.outputs1 = tf.reduce_sum(self.outputs1, axis=1, keepdims=False)
        self.outputs1 = tf.nn.l2_normalize(self.outputs1, 1)
        self.node_pred = layers.Dense(dim_mult*self.dims[-1], self.num_classes, 
                dropout=self.placeholders['dropout'],
                act=lambda x : x)
        # TF graph management
        self.node_preds = self.node_pred(self.outputs1)

        self._loss()
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var) 
                for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)
        self.preds = self.predict()

    def _loss(self):
        # Weight decay loss
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for var in self.node_pred.vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
       
        # classification loss
        if self.sigmoid_loss:
            self.loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.node_preds,
                    labels=self.placeholders['labels']))
        else:
            self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.node_preds,
                    labels=self.placeholders['labels']))

        tf.summary.scalar('loss', self.loss)

    def predict(self):
        if self.sigmoid_loss:
            return tf.nn.sigmoid(self.node_preds)
        else:
            return tf.nn.softmax(self.node_preds)

