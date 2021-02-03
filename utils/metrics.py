import tensorflow as tf

def accuracy(preds, labels):
    """Accuracy."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    return tf.reduce_sum(accuracy_all)/preds.shape[0]
