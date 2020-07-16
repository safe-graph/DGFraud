import numpy as np
import pandas as pd 
import os
from time import time
import random
import tensorflow as tf
import scipy.sparse as sp

from parse import parse_args
from get_data import Data
from model import Model

if __name__ == '__main__':
    
    args = parse_args()
    
    if args.dataset == 'dblp':
        path = "../../dataset/DBLP4057_GAT_with_idx_tra200_val_800.mat"
        save_path = "../HACUD/dblp"
        
    data_generator = Data(path=path, save_path = save_path)
    
    X_train = data_generator.X_train
    X_test = data_generator.X_test
    
    y_train = data_generator.y_train
    y_test = data_generator.y_test
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) 
    
    config = dict()
    config['n_nodes'] = data_generator.n_nodes
    config['n_metapath'] = data_generator.n_metapath
    config['n_class'] = y_train.shape[1]

    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()
    
    features = data_generator.features
    
    config['features'] =  features

    if args.adj_type == 'plain':
        config['norm_adj'] = plain_adj
        print('use the plain adjacency matrix')

    elif args.adj_type == 'norm':
        config['norm_adj'] = norm_adj
        print('use the normalized adjacency matrix')

    else:
        config['norm_adj'] = []
        for i in range(args.n_metapath):          
            config['norm_adj'].append(mean_adj[i] + sp.eye(mean_adj[i].shape[0]))
        print('use the mean adjacency matrix')

    t0 = time()

    # if args.pretrain == 1:
    #     pretrain_data = load_pretrained_data()
    # else:
    #     pretrain_data = None
    pretrain_data = None
    
    model = Model(data_config=config, pretrain_data=pretrain_data, args = args)


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    

    with tf.Session(config=config) as sess:
        
        sess.run(tf.global_variables_initializer())
        cur_best_pre_0 = 0.
        print('without pretraining.')
        
        ''' Train '''
        loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger, auc_loger = [], [], [], [], [], []
        stopping_step = 0
        should_stop = False
        
        for epoch in range(args.epoch):
            t1 = time()
            loss, ce_loss = 0., 0.
            n_batch = (data_generator.n_train-1) // args.batch_size + 1
    
            for idx in range(n_batch):
                if idx == n_batch - 1 :
                    nodes = X_train[idx*args.batch_size:]
                    labels = y_train[idx*args.batch_size:]
                else:
                    nodes = X_train[idx*int(args.batch_size):(idx+1)*int(args.batch_size)]
                    labels= y_train[idx*int(args.batch_size):(idx+1)*int(args.batch_size)]
                
                batch_loss, batch_ce_loss, reg_loss = model.train(sess, nodes, labels)
    
                loss += batch_loss
                ce_loss += batch_ce_loss
    
            test_nodes = X_test
            test_label = y_test
            
            test_loss, test_ce_loss, test_reg_loss = model.eval(sess, test_nodes, test_label)
            
            if np.isnan(loss) == True:
    
                print('ERROR: loss is nan.')
                print('ce_loss =%s' % ce_loss)
                sys.exit()
                
            log1 = 'Epoch {} Train: {:.4f} CE: {:.4f} Reg: {:.4f} Test: {:.4f}'.\
				format(epoch, loss, ce_loss, reg_loss, test_loss)
            
            print(log1)
