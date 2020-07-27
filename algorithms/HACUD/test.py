from data_loader import load_data_dblp
import scipy.io as sio
import scipy.sparse as sp
from get_data import Data
import tensorflow as tf
from sklearn import metrics

def calc_f1(y_true, y_pred):
    
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro")

def calc_auc(y_true, y_pred):
    return metrics.roc_auc_score(y_true, y_pred)

x = [1,2,0,3]