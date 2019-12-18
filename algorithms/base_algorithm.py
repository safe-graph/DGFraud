import tensorflow as tf

class Algorithm(object):
    def __init__(self, **kwargs):
        self.nodes = None

    def forward_propagation(self):
        pass

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver()
        save_path = saver.save(sess, "tmp/%s.ckpt" % 'temp')
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver()
        save_path = "tmp/%s.ckpt" % 'temp'
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)