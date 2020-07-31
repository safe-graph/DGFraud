'''
This code is due to Hengrui Zhang (@hengruizhang98) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection)
https://github.com/safe-graph/DGFraud
'''
import tensorflow as tf
import os
import sys
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


class Model(object):
    def __init__(self, data_config, pretrain_data, args):
        self.model_type = 'hacud'
        self.adj_type = args.adj_type
        self.early_stop = args.early_stop
        self.pretrain_data = pretrain_data
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        self.n_nodes = data_config['n_nodes']
        self.n_metapath = data_config['n_metapath']
        self.n_class = data_config['n_class']
        
        self.n_fold = args.n_fold
        self.n_fc = args.n_fc
        self.fc = eval(args.fc)
        self.reg = args.reg
        
    
        self.norm_adj = data_config['norm_adj']
        
        self.features = data_config['features']
        self.f_dim = self.features.shape[1]
        self.lr = args.lr
        
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        
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

        self.label = tf.placeholder(tf.float32, shape=(None, self.n_class))

        self.pred_label = self.pred(self.batch_embeddings)

        self.loss = self.create_loss(self.pred_label, self.label)
        
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
    
    def _init_weights(self):
        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()

        print('using xavier initialization')

        self.fc = [self.emb_dim] + self.fc
    
        all_weights['W'] = tf.Variable(
                initializer([self.f_dim, self.emb_dim]), name='W')
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
                initializer([self.f_dim, self.emb_dim]), name='W_rho_%d' % n)
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
                initializer([2*self.emb_dim, self.emb_dim]), name='W_f1')
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
            A_fold_hat['%d' %n] = self._split_A_hat(self.norm_adj[n])

        embeddings = self.features
        embeddings = embeddings.astype(np.float32)
        
        h = tf.matmul(embeddings, self.weights['W']) + self.weights['b']
        
        embed_u = {}
        h_u = {}
        f_u = {}
        v_u = {}
        alp_u = {}
        alp_hat = {}
        f_tilde = {}

        for n in range(self.n_metapath):

            ''' Graph Convolution '''
            embed_u['%d' %n] = [] 
            for f in range(self.n_fold):
                embed_u['%d' %n].append(tf.sparse_tensor_dense_matmul(A_fold_hat['%d' %n][f], embeddings))
            
            embed_u['%d' %n] = tf.concat(embed_u['%d' %n], 0)

            ''' Feature Fusion '''
            h_u['%d' %n] = tf.matmul(embed_u['%d' %n], self.weights['W_rho_%d' %n]) + self.weights['b_rho_%d' %n]
            f_u['%d' %n] = tf.nn.relu(tf.matmul(tf.concat([h,h_u['%d' %n]],1), self.weights['W_f_%d' %n]) 
                                            + self.weights['b_f_%d' %n])
            ''' Feature Attention '''
            v_u['%d' %n] = tf.nn.relu(tf.matmul(tf.concat([h,f_u['%d' %n]],1), self.weights['W_f1'])
                                            + self.weights['b_f1'])
            alp_u['%d' %n] = tf.nn.relu(tf.matmul(v_u['%d' %n], self.weights['W_f2'])
                                            + self.weights['b_f2'])      
            
            alp_hat['%d' %n] = tf.nn.softmax(alp_u['%d' %n], axis = 1)   

            f_tilde['%d' %n] = tf.multiply(alp_hat['%d' %n], f_u['%d' %n])                                
            
        ''' Path Attention '''

        f_c = []
        for n in range(self.n_metapath):
            f_c.append(f_tilde['%d' %n])
        f_c = tf.concat(f_c,1)

        
        for n in range(self.n_metapath):     
            if n == 0:
                beta = tf.matmul(f_c, tf.transpose(self.weights['z_%d' % n]))
                f = f_tilde['%d' %n]
                f = tf.expand_dims(f, -1)
        
            else:
                beta = tf.concat([beta, tf.matmul(f_c, tf.transpose(self.weights['z_%d' % n]))], axis = 1)
                f = tf.concat([f,tf.expand_dims(f_tilde['%d' %n],-1)], axis = 2)
        
        beta_u = tf.nn.softmax(beta, axis = 1)
        beta_u = tf.transpose(tf.expand_dims(beta_u, 0),(1,0,2))

        e_u = tf.multiply(beta_u, f)
        e_u = tf.reduce_sum(e_u, axis = 2)

        return e_u

    def pred(self, x):
        for n in range(self.n_fc):
            if n == self.n_fc - 1:
                x = tf.matmul(x, self.weights['W_%d' %n])+ self.weights['b_%d' %n]
            else:   
                x = tf.nn.relu(tf.matmul(x, self.weights['W_%d' %n])+ 
                                self.weights['b_%d' %n])
        return x

    def create_ce_loss(self, x, y):
 
        ce_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=y),0)

        return ce_loss

    def create_reg_loss(self):
        
        # for key in self.weights.keys(): 
        #     reg_loss += tf.contrib.layers.l2_regularizer(0.5)(self.weights[key])
        # regularizer = tf.contrib.layers.l2_regularizer(0.5)
        # reg_loss += tf.contrib.layers.apply_regularization(regularizer)
        reg_loss = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])

        return reg_loss

    def create_loss(self, x, y):
        self.ce_loss = self.create_ce_loss(x,y)
        self.reg_loss = self.create_reg_loss()

        loss = self.ce_loss + self.reg * self.reg_loss

        return loss

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def train(self, sess, nodes, labels):
        _, batch_loss, batch_ce_loss, reg_loss = sess.run([self.opt, self.loss, self.ce_loss, self.reg_loss],
                            feed_dict={self.nodes: nodes, self.label: labels})
        return batch_loss, batch_ce_loss, reg_loss
    
    def eval(self, sess, nodes, labels):
        loss, ce_loss, reg_loss, pred_label = sess.run([self.loss, self.ce_loss, self.reg_loss, self.pred_label],
                            feed_dict={self.nodes: nodes, self.label: labels})
        return loss, ce_loss, reg_loss, pred_label
