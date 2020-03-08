import scipy.io as sio
import scipy.sparse as sp
import numpy as np


# symmetrically normalize adjacency matrix
def normalize_adj(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).A


def construct_feed_dict(x, a, t, b, learning_rate, momentum, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['x']: x})
    feed_dict.update({placeholders['a']: a})
    feed_dict.update({placeholders['t']: t})
    feed_dict.update({placeholders['batch_index']: b})
    feed_dict.update({placeholders['lr']: learning_rate})
    feed_dict.update({placeholders['mom']: momentum})
    feed_dict.update({placeholders['num_features_nonzero']: x[1].shape})
    return feed_dict
