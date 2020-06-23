import numpy as np
from sklearn.model_selection import train_test_split
import scipy.io as sio
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
from utils.utils import pad_adjlist
import zipfile


# zip_src = '../dataset/DBLP4057_GAT_with_idx_tra200_val_800.zip'
# dst_dir = '../dataset'
def unzip_file(zip_src, dst_dir):
    iz = zipfile.is_zipfile(zip_src)
    if iz:
        zf = zipfile.ZipFile(zip_src, 'r')
        for file in zf.namelist():
            zf.extract(file, dst_dir)
    else:
        print('Zip Error.')


def load_data_dblp(path='../../dataset/DBLP4057_GAT_with_idx_tra200_val_800.mat'):
    data = sio.loadmat(path)
    truelabels, features = data['label'], data['features'].astype(float)
    N = features.shape[0]
    rownetworks = [data['net_APA'] - np.eye(N)]
    # rownetworks = [data['net_APA'] - np.eye(N), data['net_APCPA'] - np.eye(N), data['net_APTPA'] - np.eye(N)]
    y = truelabels
    index = range(len(y))
    X_train, X_test, y_train, y_test = train_test_split(index, y, stratify=y, test_size=0.4, random_state=48,
                                                        shuffle=True)

    return rownetworks, features, X_train, y_train, X_test, y_test


def load_example_semi():
    # example data for SemiGNN
    features = np.array([[1, 1, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 1, 0, 1],
                         [1, 0, 1, 1, 0, 0, 0],
                         [0, 1, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 1, 1]
                         ])
    N = features.shape[0]
    # Here we use binary matrix as adjacency matrix, weighted matrix is acceptable as well
    rownetworks = [np.array([[1, 0, 0, 1, 0, 1, 1, 1],
                             [1, 0, 0, 1, 1, 1, 0, 1],
                             [1, 0, 0, 0, 0, 0, 0, 1],
                             [0, 1, 0, 0, 1, 1, 1, 0],
                             [0, 1, 1, 1, 0, 1, 0, 0],
                             [1, 0, 0, 1, 1, 1, 0, 1],
                             [1, 0, 0, 0, 0, 0, 0, 1],
                             [0, 1, 0, 0, 1, 1, 1, 0]]),
                   np.array([[1, 0, 0, 0, 0, 1, 1, 1],
                             [0, 1, 0, 0, 1, 1, 0, 0],
                             [0, 1, 1, 1, 0, 0, 0, 0],
                             [0, 0, 1, 1, 1, 0, 0, 1],
                             [1, 1, 0, 1, 1, 0, 0, 0],
                             [1, 0, 0, 1, 0, 1, 1, 1],
                             [1, 0, 0, 1, 1, 1, 0, 1],
                             [1, 0, 0, 0, 0, 0, 0, 1]])]
    y = np.array([[0, 1], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 1]])
    index = range(len(y))
    X_train, X_test, y_train, y_test = train_test_split(index, y, stratify=y, test_size=0.2, random_state=48,
                                                        shuffle=True)  # test_size=0.25  batch——size=2

    return rownetworks, features, X_train, y_train, X_test, y_test


def load_example_gem():
    # example data for GEM
    # node=8  p=7  D=2
    features = np.array([[5, 3, 0, 1, 0, 0, 0, 1, 0],
                         [2, 3, 1, 2, 0, 0, 0, 1, 0],
                         [3, 1, 6, 4, 0, 0, 1, 1, 0],
                         [0, 0, 2, 4, 4, 1, 0, 1, 1],
                         [0, 0, 3, 3, 1, 0, 1, 0, 1],
                         [1, 2, 5, 1, 4, 1, 0, 0, 1],
                         [0, 1, 3, 5, 1, 0, 0, 0, 1],
                         [0, 3, 4, 5, 2, 1, 1, 0, 1]
                         ])
    N = features.shape[0]
    rownetworks = [np.array([[1, 1, 1, 1, 0, 0, 0, 0],
                             [1, 1, 1, 1, 0, 0, 0, 0],
                             [1, 1, 1, 1, 0, 0, 0, 0],
                             [1, 1, 1, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0]]),
                   np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 1, 1, 1, 1, 1],
                             [0, 0, 0, 1, 1, 1, 1, 1],
                             [0, 0, 0, 1, 1, 1, 1, 1],
                             [0, 0, 0, 1, 1, 1, 1, 1],
                             [0, 0, 0, 1, 1, 1, 1, 1]])]
    # y = np.array([-1, -1, -1, -1, 1, 1, 1, 1])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y = y[:, np.newaxis]
    index = range(len(y))
    X_train, X_test, y_train, y_test = train_test_split(index, y, stratify=y, test_size=0.2, random_state=8,
                                                        shuffle=True)

    return rownetworks, features, X_train, y_train, X_test, y_test


def load_data_gas():
    # example data for GAS
    # construct U-E-I network
    user_review_adj = [[0, 1], [2], [3], [5], [4, 6]]
    user_review_adj = pad_adjlist(user_review_adj)
    user_item_adj = [[0, 1], [0], [0], [2], [1, 2]]
    user_item_adj = pad_adjlist(user_item_adj)
    item_review_adj = [[0, 2, 3], [1, 4], [5, 6]]
    item_review_adj = pad_adjlist(item_review_adj)
    item_user_adj = [[0, 1, 2], [0, 4], [3, 4]]
    item_user_adj = pad_adjlist(item_user_adj)
    review_item_adj = [0, 1, 0, 0, 1, 2, 2]
    review_user_adj = [0, 0, 1, 2, 4, 3, 4]

    # initialize review_vecs
    review_vecs = np.array([[1, 0, 0, 1, 0],
                            [1, 0, 0, 1, 1],
                            [1, 0, 0, 0, 0],
                            [0, 1, 0, 0, 1],
                            [0, 1, 1, 1, 0],
                            [0, 0, 1, 1, 1],
                            [1, 1, 0, 1, 1]])

    # initialize user_vecs and item_vecs with user_review_adj and item_review_adj
    # for example, u0 has r1 and r0, then we get the first line of user_vecs: [1, 1, 0, 0, 0, 0, 0]
    user_vecs = np.array([[1, 1, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 1, 0, 1]])
    item_vecs = np.array([[1, 0, 1, 1, 0, 0, 0],
                          [0, 1, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 1, 1]])
    features = [review_vecs, user_vecs, item_vecs]

    # initialize the Comment Graph
    homo_adj = [[1, 0, 0, 0, 1, 1, 1],
                [1, 0, 0, 0, 1, 1, 0],
                [0, 0, 0, 1, 1, 1, 0],
                [1, 0, 1, 0, 0, 1, 0],
                [0, 1, 1, 1, 1, 0, 0],
                [0, 1, 1, 0, 1, 0, 0],
                [0, 1, 0, 0, 1, 0, 0]]

    adjs = [user_review_adj, user_item_adj, item_review_adj, item_user_adj, review_user_adj, review_item_adj, homo_adj]

    y = np.array([[0, 1], [1, 0], [1, 0], [0, 1], [1, 0], [1, 0], [0, 1], [1, 0]])
    index = range(len(y))
    X_train, X_test, y_train, y_test = train_test_split(index, y, stratify=y, test_size=0.4, random_state=48,
                                                        shuffle=True)

    return adjs, features, X_train, y_train, X_test, y_test
