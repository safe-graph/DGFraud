'''
This code is due to Hengrui Zhang (@hengruizhang98) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection)
https://github.com/safe-graph/DGFraud
'''
import numpy as np
import pandas as pd 
import os
from time import time
import random
import tensorflow as tf
import scipy.sparse as sp
from sklearn import metrics
from parse import parse_args
from get_data import Data
from model import Model


def calc_f1(y_true, y_pred):
    
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro")

def cal_acc(y_true, y_pred):
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    return metrics.accuracy_score(y_true, y_pred)

    # a = 0
    # b = 0
    # for i in range(len(y_true)):
    #     if y_true[i] == y_pred[i]:
    #         a+=1
    #     b+=1
    # return a/b

    
# def calc_auc(y_true, y_pred):
#     return metrics.roc_auc_score(y_true, y_pred)


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
            
            test_loss, test_ce_loss, test_reg_loss, pred_label = model.eval(sess, test_nodes, test_label)
            
            f1_scores = calc_f1(test_label, pred_label)
            acc = cal_acc(test_label, pred_label)

            # auc_score = calc_auc(pred_label, test_label)

            val_f1_mic, val_f1_mac = f1_scores[0], f1_scores[1]

            if np.isnan(loss) == True:
    
                print('ERROR: loss is nan.')
                print('ce_loss =%s' % ce_loss)
                sys.exit()
                
            log1 = 'Epoch {} Train: {:.4f} CE: {:.4f} Reg: {:.4f} Test: {:.4f} F1_mic: {:.4f} F1_mac: {:.4f} Accuracy: {:.4f}'.\
				format(epoch, loss, ce_loss, reg_loss, test_loss, val_f1_mic, val_f1_mac, acc)

            print(log1)
