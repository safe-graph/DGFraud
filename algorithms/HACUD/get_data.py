'''
This code is due to Hengrui Zhang (@hengruizhang98) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection)
https://github.com/safe-graph/DGFraud
'''
import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
from data_loader import load_data_dblp
import os

class Data(object):
    def __init__(self, path, save_path):
        self.path = path
        self.save_path = save_path
        
        self.rownetworks, self.features, self.X_train, self.y_train, self.X_test, self.y_test = load_data_dblp(path)
        self.n_nodes = 0
        self.n_train, self.n_test = 0, 0
        
        self.n_nodes = len(self.features)
        self.n_train = len(self.X_train)
        self.n_test = len(self.X_test)
        
        self.n_metapath = len(self.rownetworks)
        adj = []
        u_index = []
        v_index = []
        self.n_int = []

        for i in range(self.n_metapath):
            z = self.rownetworks[i]
            adj.append(z)
            u_index.append(np.where(z)[0])
            v_index.append(np.where(z)[1])   
            self.n_int.append(len(np.where(z)[0]))
        
        self.print_statistics()

        self.R = []
        for i in range(self.n_metapath):
            R = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
            R[u_index[i], v_index[i]] = 1
            self.R.append(R)
        
 
    def get_adj_mat(self):
        
        try:
            t1 = time()
            adj_mat = []
            norm_adj_mat = []
            mean_adj_mat = []

            for i in range(self.n_metapath):
                adj = sp.load_npz(self.save_path + '/s_adj_%d_mat.npz' %i)
                norm = sp.load_npz(self.save_path + '/s_norm_adj_%d_mat.npz' %i)
                mean = sp.load_npz(self.save_path + '/s_mean_adj_%d_mat.npz' %i)

                adj_mat.append(adj)
                norm_adj_mat.append(norm)
                mean_adj_mat.append(mean)

            print('already load adj matrix', adj_mat[0].shape, time() - t1)

        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            for i in range(self.n_metapath):
                sp.save_npz(self.save_path + '/s_adj_%d_mat.npz' %i, adj_mat[i])
                sp.save_npz(self.save_path + '/s_norm_adj_%d_mat.npz' %i, norm_adj_mat[i])
                sp.save_npz(self.save_path + '/s_mean_adj_%d_mat.npz' %i, mean_adj_mat[i])
                
        return adj_mat, norm_adj_mat, mean_adj_mat

    def create_adj_mat(self):
        
        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
     
            return norm_adj.tocoo()

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp
        
        _adj = []
        norm_adj = []
        mean_adj = []
        for i in range(self.n_metapath):
            print('metapath', i)
            t1 = time()
            adj_mat = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = self.R[i].tolil()

            adj_mat[:self.n_nodes,:self.n_nodes] = R
            adj_mat = adj_mat.todok()
            print('already create adjacency matrix', adj_mat.shape, time() - t1)

            t2 = time()

            norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
            mean_adj_mat = normalized_adj_single(adj_mat)

            print('already normalize adjacency matrix', time() - t2)

            _adj.append(adj_mat.tocsr())
            norm_adj.append(norm_adj_mat.tocsr())
            mean_adj.append(mean_adj_mat.tocsr())
        return _adj, norm_adj, mean_adj





    def print_statistics(self):
        print('n_metapaths=%d' % (self.n_metapath))
        print('n_metapahts=%d' % (self.n_metapath))
        print('n_nodes=%d' % (self.n_nodes))
        print('n_interactions=%s' % (self.n_int))
        print('n_train=%d, n_test=%d, sparsity=%s' % (self.n_train, self.n_test, (np.array(self.n_int)/(self.n_nodes * self.n_nodes))))


    def get_sparsity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open(self.save_path + '/sparsity.split', 'r').readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split(' ')])
            print('get sparsity split.')

        except Exception:
            split_uids, split_state = self.create_sparsity_split()
            f = open(self.save_path + '/sparsity.split', 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create sparsity split.')

        return split_uids, split_state



    def create_sparsity_split(self):
        all_users_to_test = list(self.test_set.keys())
        user_n_iid = dict()

        # generate a dictionary to store (key=n_iids, value=a list of uid).
        for uid in all_users_to_test:
            train_iids = self.train_items[uid]
            test_iids = self.test_set[uid]

            n_iids = len(train_iids) + len(test_iids)

            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)
        split_uids = list()

        # split the whole user set into four subset.
        temp = []
        count = 1
        fold = 4
        n_count = (self.n_train + self.n_test)
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' %(n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

        return split_uids, split_state

# if __name__ == '__main__':
#     path = "../../dataset/DBLP4057_GAT_with_idx_tra200_val_800.mat"
#     data_generator = Data(path=path, save_path = save_path)
