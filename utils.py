import numpy as np
import pickle
from scipy.sparse import csr_matrix
from scipy import sparse


def train_test_split(sparse_mat, test_size=0.1):

    nnz_row_inds, nnz_col_inds = sparse_mat.nonzero()
    nnz_num = sparse_mat.count_nonzero()

    test_nnz_num = int(np.round(test_size * nnz_num))

    rng = np.random.default_rng(seed=42)
    test_nnz_inds = rng.choice(np.arange(nnz_num), test_nnz_num, replace=False)
    test_nnz_row_inds, test_nnz_col_inds = nnz_row_inds[test_nnz_inds], nnz_col_inds[test_nnz_inds]

    test_data = csr_matrix((sparse_mat[test_nnz_row_inds, test_nnz_col_inds].A[0], (test_nnz_row_inds, test_nnz_col_inds)), shape=sparse_mat.shape)
    train_data = sparse_mat - test_data

    return train_data, test_data


def weight_mat(train_data, weight_type):

    if weight_type == 1:
        train_nnz_row_index, train_nnz_col_index = train_data.nonzero()
        weight_mat = np.zeros(shape=train_data.shape)
        weight_mat[train_nnz_row_index, train_nnz_col_index] = 1

    elif weight_type == 2:
        row_l2_norm = np.linalg.norm(train_data.A, axis=1)
        col_l2_vec2mat = np.tile(row_l2_norm, (train_data.shape[0], 1))
        col_l2_norm = np.nan_to_num(train_data / col_l2_vec2mat.transpose())
        weight_mat = np.matmul(col_l2_norm, col_l2_norm.transpose())
        weight_mat = np.array(weight_mat)

    elif weight_type == 3:
        weight_mat = np.ones(shape=train_data.shape)

    elif weight_type == 4:
        col_l2_norm = np.linalg.norm(train_data.A, axis=0)
        col_l2_norm_vec2mat = np.tile(col_l2_norm, (train_data.shape[0], 1))
        col_normalization = np.nan_to_num(train_data / col_l2_norm_vec2mat)
        weight_mat = np.matmul(col_normalization.transpose(), col_normalization)
        weight_mat = np.array(weight_mat)

    return weight_mat

def Txt2SpaMat(txtfile):

    "Read .txt file to sparse matrix."

    txt_array = np.loadtxt(txtfile, dtype=int)
    edge_num = txt_array.shape[0]
    row_index = txt_array[:, 0]
    col_index = txt_array[:, 1]
    val = np.ones(edge_num, dtype=int)

    mat_dim = txt_array.max() + 1
    spa_mat = sparse.csc_array((val, (row_index, col_index)), shape=(mat_dim, mat_dim))
    return spa_mat


def Remove_MultipleEdges_SelfLoop_Directed(txtfile):

    "Remove multiple edges and self-connections from a directed network."

    spa_mat = Txt2SpaMat(txtfile)
    print("before:", spa_mat.count_nonzero())
    row_indexes_list, col_indexes_list = spa_mat.nonzero()
    non_zeros_ele_num = spa_mat.count_nonzero()
    # for i in range(non_zeros_ele_num):
    #     row_index = row_indexes_list[i]
    #     column_index = col_indexes_list[i]
    #     if spa_mat[row_index, column_index] == spa_mat[column_index, row_index]:
    #         spa_mat[row_index, column_index] = 1
    #         spa_mat[column_index, row_index] = 0

    row_num, col_num = spa_mat.shape
    zero_array = np.zeros(row_num, dtype=int)
    spa_mat.setdiag(zero_array)

    return spa_mat


def SimpleNetwork2SpaMat(txtfile : str, head_skip_rows : int):

    """
    Convert a simple network to a sparse matrix, index start with 1.
    """
    txt_array = np.loadtxt(txtfile, dtype=int, skiprows=head_skip_rows)
    mat_dim = txt_array.max()

    txt_arr = txt_array - 1
    row_index_array = txt_arr[:,0]
    col_index_array = txt_arr[:,1]

    edge_num = txt_array.shape[0]
    val = np.ones(edge_num, dtype=float)

    spa_mat = sparse.csc_matrix((val, (row_index_array, col_index_array)), shape=(mat_dim, mat_dim))

    return spa_mat


def UniformDrawSamples(row_num, col_num, sampling_ratio):

    whole_samples = [(i,j) for i in range(row_num) for j in range(col_num)]
    samples_num = int(np.round(row_num * col_num * sampling_ratio))
    rng = np.random.default_rng(seed=42)
    samples = rng.choice(whole_samples, samples_num, replace=False)
    return samples


def ColNor(adj_mat):

    row_num, col_num = adj_mat.shape
    col_sum = np.sum(adj_mat, axis=0)
    mat_col_norm = np.nan_to_num(np.divide(adj_mat.A, col_sum))
    return mat_col_norm


def SimpleNetwork2SpaMat_0(txtfile : str, head_skip_rows : int):

    txt_array = np.loadtxt(txtfile, dtype=int, skiprows=head_skip_rows)
    edge_num = txt_array.shape[0]

    mat_dim = txt_array.max() + 1
    row_index_array = txt_array[:, 0]
    col_index_array = txt_array[:, 1]
    val = np.ones(edge_num, dtype=float)

    spa_mat = sparse.csc_matrix((val, (row_index_array, col_index_array)), shape=(mat_dim, mat_dim))
    return spa_mat

