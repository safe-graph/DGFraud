"""
This code is due to Yutong Deng (@yutongD), Yingtong Dou (@YingtongDou) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection)
https://github.com/safe-graph/DGFraud
"""
import tensorflow as tf
from base_models.layers import MeanAggregator

init_fn = tf.keras.initializers.GlorotUniform

class GraphSage(tf.keras.Model):
	"""
	GraphSage base model outputing embeddings of given nodes
	"""
	def __init__(self, raw_features, internal_dim, num_layers, num_classes):

		super().__init__()
		self.input_layer = RawFeature(raw_features, name="raw_feature_layer")
		self.seq_layers = []
		for i in range (1, num_layers + 1):
			layer_name = "agg_lv" + str(i)
			input_dim = internal_dim if i > 1 else raw_features.shape[-1]
			aggregator_layer = MeanAggregator(input_dim, internal_dim, name=layer_name, activ = True)
			self.seq_layers.append(aggregator_layer)
		
		self.classifier = tf.keras.layers.Dense(num_classes,
												activation = tf.nn.softmax,
												use_bias = False,
												kernel_initializer = init_fn,
												name = "classifier",
												)

	def call(self, minibatch):
		"""
		:param [node] nodes: target nodes for embedding
		"""
		x = self.input_layer(tf.squeeze(minibatch.src_nodes))
		for aggregator_layer in self.seq_layers:
			x = aggregator_layer(x,
								minibatch.dstsrc2srcs.pop(),
								minibatch.dstsrc2dsts.pop(), 
								minibatch.dif_mats.pop()
								)
		return self.classifier(x)

class RawFeature(tf.keras.layers.Layer):
	def __init__(self, features, **kwargs):
		"""
		:param ndarray((#(node), #(feature))) features: a matrix, each row is feature for a node
		"""
		super().__init__(trainable=False, **kwargs)
		self.features = tf.constant(features)
		
	def call(self, nodes):
		"""
		:param [int] nodes: node ids
		"""
		return tf.gather(self.features, nodes)
