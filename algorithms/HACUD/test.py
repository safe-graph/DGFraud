from data_loader import load_data_dblp
import scipy.io as sio
import scipy.sparse as sp
from get_data import Data
<<<<<<< HEAD
import tensorflow as tf
=======
>>>>>>> 0cdf7c11220f22aca01305a8da6b3376d92b6298

path='../../dataset/DBLP4057_GAT_with_idx_tra200_val_800.mat'
save_path = "../HACUD/dblp"

# rownetworks, features, X_train, y_train, X_test, y_test = load_data_dblp(path)

# data = sio.loadmat(path)
# truelabels, features = data['label'], data['features'].astype(float)
# N = features.shape[0]

<<<<<<< HEAD
# data_generator = Data(path=path, save_path = save_path)
=======
data_generator = Data(path=path, save_path = save_path)
>>>>>>> 0cdf7c11220f22aca01305a8da6b3376d92b6298

# print('n_train=%d, n_test=%d, sparsity=%.s' % (self.n_train, self.n_test, (np.array(self.n_int)/(100 * 100))))

# adj_mat, norm_adj_mat, mean_adj_mat = data_generator.create_adj_mat()
<<<<<<< HEAD
# sp.save_npz(save_path + '/s_adj_0_mat.npz', adj_mat[0])
=======
# sp.save_npz(save_path + '/s_adj_0_mat.npz', adj_mat[0])
>>>>>>> 0cdf7c11220f22aca01305a8da6b3376d92b6298
