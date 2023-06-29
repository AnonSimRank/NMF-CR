import numpy as np
from scipy import sparse
from utils import SimpleNetwork2SpaMat_0, train_test_split, SimpleNetwork2SpaMat
from NMF_CR import DecomAdjMat
import Metrics
import os

# ############################### Read sparse matrix ###################################################################
txt_file = r"dataset/ASongIceFire.txt"

save_file_dir = r"output/NMF_C/k_256"
if not os.path.exists(save_file_dir):
    os.makedirs(save_file_dir)

Spa_mat = SimpleNetwork2SpaMat(txt_file, 0)

##### undirected graph ###################################
Spa_mat_update = Spa_mat + Spa_mat.transpose()
Spa_mat_update_file = save_file_dir + "/" + "Spa_mat_update.txt"
np.savetxt(Spa_mat_update_file, Spa_mat_update.A)

##############################################################
# ########################################################## Divide into train set and test set #######################
train_data, test_data = train_test_split(Spa_mat_update, 0.1)
train_data_file = save_file_dir + "/" + "train_data.npz"
sparse.save_npz(train_data_file, train_data)

test_data_file = save_file_dir + "/" + "test_data.npz"
sparse.save_npz(test_data_file, test_data)

# ####################################################################################################################
# ####################################### weight matrix ##############################################################
train_nnz_row_index, train_nnz_col_index = train_data.nonzero()
weight_mat = np.zeros(shape=Spa_mat_update.shape)
weight_mat[train_nnz_row_index, train_nnz_col_index] = 1

col_l2_norm = np.linalg.norm(train_data.A, axis=0)
col_l2_norm_vec2mat = np.tile(col_l2_norm, (Spa_mat_update.shape[0], 1))
col_normalization = np.nan_to_num(train_data / col_l2_norm_vec2mat)

# ########################################################################################################################
# ############################################## train ################################################################
lambda_0 = 0.1
lambda_1 = 0.1
max_iter_num = 20000
k = 256
thre_val = 1.0e-8
U, V, C= DecomAdjMat(col_normalization, weight_mat, lambda_0, lambda_1, max_iter_num, k, thre_val)
U_dir = save_file_dir + "/" + "U.txt"
V_dir = save_file_dir + "/" + "V.txt"
C_dir = save_file_dir + "/" + "C.txt"
np.savetxt(U_dir, U, fmt='%6f', delimiter=' ')
np.savetxt(V_dir, V, fmt='%6f', delimiter=' ')
np.savetxt(C_dir, C, fmt='%6f', delimiter=' ')
# ############################################## finish train #########################################################
# ############################################### eval ###############################################################
pred_data = np.matmul(U, V.transpose())
pred_data = np.multiply(pred_data, col_l2_norm_vec2mat)
pred_data_file = save_file_dir + "/" + "pred_data.txt"
np.savetxt(pred_data_file, pred_data)

precision_score = Metrics.precision(train_data, test_data, pred_data)
print("precision:", precision_score)
precision_score_file = save_file_dir + "/" + "precision_score.txt"
np.savetxt(precision_score_file, np.array([precision_score]))

recall_score = Metrics.recall(train_data, test_data, pred_data)
print("recall:", recall_score)
recall_score_file = save_file_dir + "/" + "recall_score.txt"
np.savetxt(recall_score_file, np.array([recall_score]))

f_score = Metrics.F_score(precision_score, recall_score)
print("f_score:", f_score)
f_score_file = save_file_dir + "/" + "f_score.txt"
np.savetxt(f_score_file, np.array([f_score]))

auc_score = Metrics.AUC(train_data, test_data, pred_data, n=1000)
print("auc_score:", auc_score)
auc_score_file = save_file_dir + "/" + "auc_score.txt"
np.savetxt(auc_score_file, np.array([auc_score]))

# #####################################################################################################################

