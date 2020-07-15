import tensorflow as tf
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from utility.helper import *
from utility.batch_test import *

class HACUD(object):
    def __init__(self, data_config, pretrain_data, args):
        self.model_type = 'hacud'
        self.adj_type = args.adj_type
        self.early_stop = args.early_stop
        self.pretrain_data = pretrain_data
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        self.n_node = data_config['n_node']
        self.n_fold = args.n_fold
        self.n_fc = args.n_fc
        self.fc = args.fc
        self.reg = args.reg
        self.n_metapath = data_config['n_metapath']
        self.norm_adj = {}
        for n in range(self.n_metapath):
            self.norm_adj['%d' %n] = data_config['norm_adj_%d' %n]
        
        self.lr = args.lr
        
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        
        self.decay = self.regs[0]

        self.verbose = args.verbose

        '''
        Create Placeholder for Input Data & Dropout.
        '''
        # placeholder definition
        self.nodes = tf.placeholder(tf.int32, shape=(None,))

        '''
        Create Model Parameters (i.e., Initialize Weights).
        '''
        # initialization of model parameters
        self.weights = self._init_weights()

        '''
        Compute Graph-based Representations of all nodes
        '''
        self.n_embeddings = self._create_embedding()

        '''
        Establish the representations of nodes in a batch.
        '''
        self.batch_embeddings = tf.nn.embedding_lookup(self.n_embeddings, self.nodes)

        self.label = tf.placeholder(tf.int32, shape=(None,))

        self.loss = self.create_loss(self.batch_embeddings, loss)
        
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
    
    def _init_weights(self):
        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()

        if self.pretrain_data is None:
            all_weights['node_embedding'] = tf.Variable(initializer([self.n_nodes, self.emb_dim]), name='node_embedding')
            print('using xavier initialization')
        else:
            all_weights['node_embedding'] = tf.Variable(initial_value=self.pretrain_data['node_embed'], trainable=True,
                                                        name='node_embedding', dtype=tf.float32)
            print('using pretrained initialization')

        self.weight_size_list = [self.emb_dim] + self.weight_size

        tf.add_to_collection(tf.GraphKeys.WEIGHTS, all_weights['node_embedding'])

            
        
        all_weights['W'] = tf.Variable(
                initializer([self.emb_dim, self.emb_dim]), name='W')
        all_weights['b'] = tf.Variable(
                initializer([1, self.emb_dim]), name='b')
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, all_weights['W'])
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, all_weights['b'])  

        for n in range(self.n_fc):
            all_weights['W_%d' % n] = tf.Variable(
                    initializer([self.fc[n], self.fc[n+1]]), name='W_%d' % n)
            all_weights['b_%d' % n] = tf.Variable(
                    initializer([1, self.fc[n+1]]), name='b_%d' % n)
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, all_weights['W_%d' %n])
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, all_weights['b_%d' %n])

        for n in range(self.n_metapath):
            all_weights['W_rho_%d' % n] = tf.Variable(
                initializer([self.emb_dim, self.emb_dim]), name='W_rho_%d' % n)
            all_weights['b_rho_%d' % n] = tf.Variable(
                initializer([1, self.emb_dim]), name='b_rho_%d' % n)
            all_weights['W_f_%d' % n] = tf.Variable(
                initializer([2*self.emb_dim, self.emb_dim]), name='W_f_%d' % n)
            all_weights['b_f_%d' % n] = tf.Variable(
                initializer([1, self.emb_dim]), name='b_f_%d' % n) 
            all_weights['z_%d' % n] = tf.Variable(
                initializer([1, self.emb_dim*self.n_metapath]), name='z_%d' % n) 

            tf.add_to_collection(tf.GraphKeys.WEIGHTS, all_weights['W_rho_%d' %n])
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, all_weights['b_rho_%d' %n])
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, all_weights['W_f_%d' %n])
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, all_weights['b_f_%d' %n])
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, all_weights['z_%d' %n])

        all_weights['W_f1'] = tf.Variable(
                initializer([self.emb_dim, self.emb_dim]), name='W_f1')
        all_weights['b_f1'] = tf.Variable(
                initializer([1, self.emb_dim]), name='b_f1')  
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, all_weights['W_f1'])
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, all_weights['b_f1'])

        all_weights['W_f2'] = tf.Variable(
                initializer([self.emb_dim, self.emb_dim]), name='W_f2')
        all_weights['b_f2'] = tf.Variable(
                initializer([1, self.emb_dim]), name='b_f2') 
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, all_weights['W_f2'])
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, all_weights['b_f2'])

        return all_weights       

    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.n_nodes) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_nodes
            else:
                end = (i_fold + 1) * fold_len
            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
            
        return A_fold_hat
    
    def _create_embedding(self):
        
        A_fold_hat = {}
        for n in range(self.n_metapath):
            A_fold_hat['%d' %n] = self._split_A_hat(self.norm_adj['%d' %n])
        embeddings = self.weights['node_embedding']
        
        h = tf.matmul(embeddings, self.weights['W']) + self.weights['b']
        
        embed_u = {}
        h_u = {}
        f_u = {}
        v_u = {}
        alp_u = {}
        alp_hat = {}
        f_tilde = {}

        for n in range(self.n_metapath):
            embed_u['%d' %n] = [] 
            for f in range(self.n_fold):
                embed_u['%d' %n].append(tf.sparse_tensor_dense_matmul(A_fold_hat['%d' %n][f], embeddings))
            
            embed_u['%d' %n] = tf.concat(embed_u['%d' %n], 0)

            h_u['%d' %n] = tf.matmul(embed_u['%d' %n], self.weights['W_rho_%d' %n]) + self.weights['b_rho_%d' %n]
            f_u['%d' %n] = tf.nn.relu(tf.matmul(tf.concat([h,h_u['%d' %n]],1), self.weights['W_f_%d' %n]) 
                                            + self.weights['b_f_%d' %n])
            ''' Feature Attention '''
            v_u['%d' %n] = tf.nn.relu(tf.matmul(tf.concat([h,f_u['%d' %n]],1), self.weights['W_f1'])
                                            + self.weights['b_f1'])
            alp_u['%d' %n] = tf.nn.relu(tf.matmul(v_u['%d' %n], self.weights['W_f2'])
                                            + self.weights['b_f2'])      
            alp_hat['%d' %n] = tf.nn.softmax(alp_hat, axis = 1)   

            f_tilde['%d' %n] = tf.multiply(alp_hat['%d' %n], f_u['%d' %n])                                
            
        ''' Path Attention '''

        f_c = []
        for n in range(self.n_metapath):
            f_c.append(f_tilde['%d' %n])
        f_c = tf.concat(f_c,0)

        
        for n in range(self.n_metapath):     
            if n == 0:
                beta = tf.matmul(f_c, tf.transpose(self.weights['z_%d' % n]))
                f = f_tilde['%d' %n]
            else:
                beta = tf.concat([beta, tf.matmul(f_c, tf.transpose(self.weights['z_%d' % n]))], axis = 1)
                f = tf.concat([f,f_tilde['%d' %n]], axis = 2)
        
        beta_u = tf.nn.softmax(beta, axis = 1)
        beta_u = tf.transpose(tf.expand_dims(beta_u, 0),(1,0,2))

        e_u = tf.multiply(beta_u, f)
        e_u = tf.reduce_sum(e_u, axis = 2)

        return e_u

    def create_ce_loss(self, x, y):
        for n in range(self.n_fc):
            if n == self.n_fc - 1:
                x = tf.matmul(x, self.weights['W_%d' %n])+ self.weights['b_%d' %n]
            else:   
                x = tf.nn.relu(tf.matmul(x, self.weights['W_%d' %n])+ 
                                self.weights['b_%d' %n])
        ce_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y),1)

        return ce_loss

    def create_reg_loss(self):
        reg_loss = 0 
        regularizer = tf.contrib.layers.l2_regularizer(1)
        reg_loss += tf.contrib.layers.apply_regularization(regularizer)

        return reg_loss

    def create_loss(self, x, y):
        ce_loss = self.create_ce_loss(x,y)
        reg_loss = self.create_reg_loss

        loss = ce_loss + self.reg*reg_loss

        return loss