"""
This code is due to Kay Liu (@kayzliu), Yingtong Dou (@YingtongDou) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection)
https://github.com/safe-graph/DGFraud
"""
import tensorflow as tf
from base_models.layers import SageMeanAggregator

init_fn = tf.keras.initializers.GlorotUniform


class GraphSage(tf.keras.Model):
	"""
	GraphSage base model outputing embeddings of given nodes
	"""

	def __init__(self, features_dim, internal_dim, num_layers, num_classes):
		"""
		:param int features_dim: input dimension
		:param int internal_dim: hidden layer dimension
		:param int num_layers: number of sample layer
		:param int num_classes: number of node classes
		"""
		super().__init__()
		self.seq_layers = []
		for i in range(1, num_layers + 1):
			layer_name = "agg_lv" + str(i)
			input_dim = internal_dim if i > 1 else features_dim
			aggregator_layer = SageMeanAggregator(input_dim, internal_dim, name=layer_name, activ=True)
			self.seq_layers.append(aggregator_layer)

		self.classifier = tf.keras.layers.Dense(num_classes,
												activation=tf.nn.softmax,
												use_bias=False,
												kernel_initializer=init_fn,
												name="classifier",
												)

	def call(self, minibatch, features):
		"""
		:param namedtuple minibatch: minibatch of target nodes
		:param tensor features: 2d features of nodes
		"""
		x = tf.gather(tf.constant(features, dtype=float), tf.squeeze(minibatch.src_nodes))
		for aggregator_layer in self.seq_layers:
			x = aggregator_layer(x,
								 minibatch.dstsrc2srcs.pop(),
								 minibatch.dstsrc2dsts.pop(),
								 minibatch.dif_mats.pop()
								 )
		return self.classifier(x)
