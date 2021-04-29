"""
This code is due to Kay Liu (@kayzliu), Yingtong Dou (@YingtongDou) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection)
https://github.com/safe-graph/DGFraud
"""
import tensorflow as tf
from base_models.layers import ConsisMeanAggregator

init_fn = tf.keras.initializers.GlorotUniform


class GraphConsis(tf.keras.Model):
	"""
	GraphConsis base model outputing embeddings of given nodes
	"""

	def __init__(self, features_dim, internal_dim, num_layers, num_classes, num_relations):
		"""
		:param int features_dim: input dimension
		:param int internal_dim: hidden layer dimension
		:param int num_layers: number of sample layer
		:param int num_classes: number of node classes
		:param int num_relations: number of relations
		"""
		super().__init__()
		self.seq_layers = []
		self.attention_vec = tf.Variable(tf.random.uniform([2 * internal_dim, 1], dtype=tf.float32))
		self.relation_vectors = tf.Variable(tf.random.uniform([num_relations, internal_dim], dtype=tf.float32))
		for i in range(1, num_layers + 1):
			layer_name = "agg_lv" + str(i)
			input_dim = internal_dim if i > 1 else features_dim
			aggregator_layer = ConsisMeanAggregator(input_dim, internal_dim, name=layer_name)
			self.seq_layers.append(aggregator_layer)

		self.classifier = tf.keras.layers.Dense(num_classes,
												activation=tf.nn.softmax,
												use_bias=False,
												kernel_initializer=init_fn,
												name="classifier",
												)

	def call(self, minibatchs, features):
		"""
		:param [namedtuple] minibatchs: minibatch list of each relation
		:param tensor features: 2d features of nodes
		"""
		xs = []
		for i, minibatch in enumerate(minibatchs):
			x = tf.gather(tf.Variable(features, dtype=float), tf.squeeze(minibatch.src_nodes))
			for aggregator_layer in self.seq_layers:
				x = aggregator_layer(x,
									 minibatch.dstsrc2srcs.pop(),
									 minibatch.dstsrc2dsts.pop(),
									 minibatch.dif_mats.pop(),
									 tf.nn.embedding_lookup(self.relation_vectors,i),
									 self.attention_vec
									 )
			xs.append(x)

		return self.classifier(tf.nn.l2_normalize(tf.reduce_sum(tf.stack(xs, 1), axis=1, keepdims=False), 1))
