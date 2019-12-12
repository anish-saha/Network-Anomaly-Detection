import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import confusion_matrix


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def load_data(path="../data/twitter/", dataset="twitter"):
    """Load Twitter dataset"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.features_mid".format(path, dataset),
                                        delimiter=',',
                                        dtype=np.dtype(str))

    # delete rows and columns that are not required
    idx_features_labels = np.delete(idx_features_labels, np.s_[0], axis=0)
    idx_features_labels = np.delete(idx_features_labels, np.s_[1:3], axis=1)

    # shuffle the rows
    np.random.shuffle(idx_features_labels)

    features = sp.csr_matrix(idx_features_labels[:, 2:], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, 1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.graph_mid".format(path, dataset),
                                    delimiter=',',
                                    dtype=np.int32)

    # delete rows that we don't need
    edges_unordered = np.delete(edges_unordered, np.s_[0], axis=0)

    x = list(map(idx_map.get, edges_unordered.flatten()))
    y = [0 if i is None else i for i in x]
    edges = np.array(y, dtype=np.int32)
    edges = edges.reshape(edges_unordered.shape)
    print("edges shape:", edges.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = torch.FloatTensor(np.array(adj.todense()))

    idx_train = list(range(0, 4000))
    idx_val = list(range(4000, 4500))
    idx_test = list(range(4500, 5000))

    np.random.shuffle(idx_train)
    np.random.shuffle(idx_val)
    np.random.shuffle(idx_test)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    acc = correct / len(labels)
    confusion_m = confusion_matrix(labels, preds).ravel()
    return (acc, confusion_m)

def accuracy_tw(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    print("total labels:", len(labels))
    print("total preds:", len(preds))
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    print("confusion matrix (tn,fp,fn,tp):", tn, fp, fn, tp)
    print("Accuracy:", correct / len(labels))
    print("Precision:", tp/(tp+fp))
    print("Recall:", tp / (tp + fn))
    return correct / len(labels)
