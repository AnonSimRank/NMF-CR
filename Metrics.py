import numpy as np
from sklearn.metrics import precision_recall_curve

def precision(A_train, A_probe, A_pred):

    non_observed = (A_train.A == 0)
    non_observed_true = (A_probe.A[non_observed] != 0).astype('int')
    # non_observed_true = A_probe[non_observed]

    non_observed_pred = A_pred[non_observed]

    order = non_observed_pred.argsort()[::-1]
    L = (A_probe != 0).sum()

    precision_score = non_observed_true[order][:L].mean()
    return precision_score


def recall(A_train, A_probe, A_pred):

    non_observed = (A_train.A == 0)
    non_observed_true = (A_probe.A[non_observed] > 0).astype('int')

    non_observed_pred = A_pred[non_observed]

    order = non_observed_pred.argsort()[::-1]

    L = (A_probe != 0).sum()

    recall_score = non_observed_true[order][:L].sum() / non_observed_true.sum()

    return recall_score


def F_score(precision_score, recall_score):

    f_score = 2 * precision_score * recall_score / (precision_score + recall_score)

    return f_score


def pr_curve(A_train, A_probe, A_pred):

    non_observed = (A_train.A == 0)
    non_observed_true = (A_probe.A[non_observed] > 0).astype('int')
    non_observed_pred = A_pred[non_observed]

    precision, recall, _ = precision_recall_curve(non_observed_true, non_observed_pred)
    return recall, precision


def AUC(A_train, A_probe, A_pred, n=100):

    A_shape = A_train.shape[0]
    missing_links_idx = A_probe.nonzero()
    missing_links_idx = missing_links_idx[0] * A_shape + missing_links_idx[1]

    nonzero_full_idx = (A_train + A_probe).nonzero()
    nonzero_full_idx = nonzero_full_idx[0] * A_shape + nonzero_full_idx[1]

    nonobserved_links_idx = np.setdiff1d(np.arange(A_shape**2), nonzero_full_idx)

    np.random.seed(0)
    missing_links_idx_n = np.random.choice(missing_links_idx, n)
    np.random.seed(0)
    nonobserved_links_idx_n = np.random.choice(nonobserved_links_idx, n)

    missing_ele_weights = A_pred[missing_links_idx_n // A_shape, missing_links_idx_n % A_shape]
    nonobserved_ele_weights = A_pred[nonobserved_links_idx_n // A_shape, nonobserved_links_idx_n % A_shape]

    n1 = np.sum(missing_ele_weights > nonobserved_ele_weights)
    n2 = np.sum(missing_ele_weights == nonobserved_ele_weights)

    auc_score = (n1 + 0.5 * n2) / n

    return auc_score
