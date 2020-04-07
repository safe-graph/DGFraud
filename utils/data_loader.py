import numpy as np
from sklearn.model_selection import train_test_split
import scipy.io as sio
import yaml
from utils.utils import pad_adjlist


def read_data_dzdp():
    index = list(range(9067))
    y = np.loadtxt('data/label.txt')
    X_train, X_test, y_train, y_test = train_test_split(index, y, stratify=y, test_size=0.4,
                                                        random_state=48, shuffle=True)

    return X_train, y_train, X_test, y_test


def load_data_dblp(path='data/DBLP4057_GAT_with_idx_tra200_val_800.mat'):
    data = sio.loadmat(path)
    truelabels, features = data['label'], data['features'].astype(float)
    N = features.shape[0]
    rownetworks = [data['net_APCPA'] - np.eye(N),data['net_APA'] - np.eye(N)]
    # rownetworks = [data['net_APA'] - np.eye(N), data['net_APCPA'] - np.eye(N), data['net_APTPA'] - np.eye(N)]
    y = truelabels
    index = range(len(y))
    X_train, X_test, y_train, y_test = train_test_split(index, y, stratify=y, test_size=0.4, random_state=48,
                                                        shuffle=True)

    return rownetworks, features, X_train, y_train, X_test, y_test


def load_example_semi():
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
                                                        shuffle=True)# test_size=0.25  batch——size=2

    return rownetworks, features, X_train, y_train, X_test, y_test


def load_data_yelp():
    # init vectors of review, user and item
    review_vecs = np.loadtxt('data/yelp_data/yelp_review_tfidf.txt')
    user_vecs = np.loadtxt('data/yelp_data/user_review_adj.txt')
    item_vecs = np.loadtxt('data/yelp_data/item_review_adj.txt')
    features = [review_vecs, user_vecs, item_vecs]

    # init adj
    file1 = open('data/yelp_data/user_review_adjlist.yaml', 'r')
    user_review_adj = yaml.load(file1)
    file2 = open('data/yelp_data/user_review_item_adjlist.yaml', 'r')
    review_item_adj = yaml.load(file2)
    file3 = open('data/yelp_data/item_review_adjlist.yaml', 'r')
    item_review_adj = yaml.load(file3)
    file4 = open('data/yelp_data/item_review_user_adjlist.yaml', 'r')
    review_user_adj = yaml.load(file4)
    adjs = [user_review_adj, review_item_adj, item_review_adj, review_user_adj]

    y = np.loadtxt('data/yelp_data/review_label.txt')
    index = range(len(y))
    X_train, X_test, y_train, y_test = train_test_split(index, y, stratify=y, test_size=0.4, random_state=48,
                                                        shuffle=True)

    return adjs, features, X_train, y_train, X_test, y_test


def load_data_example():
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
