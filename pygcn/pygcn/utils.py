import numpy as np
import scipy.sparse as sp
import torch

from sklearn.metrics import confusion_matrix


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/twitter/", dataset="twitter"):
    """Load filtered Twitter dataset"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.features".format(path, dataset),
                                        delimiter=',',
                                        dtype=np.dtype(str))

    # delete rows and columns that are not required
    idx_features_labels = np.delete(idx_features_labels, np.s_[0], axis=0)
    idx_features_labels = np.delete(idx_features_labels, np.s_[1:3], axis=1)

    # shuffle the rows
    np.random.shuffle(idx_features_labels)

    # delete columns that are not required
    features = sp.csr_matrix(idx_features_labels[:, 2:], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, 1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.graph".format(path, dataset),
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

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    # total size is 75624 entries
    # train size: 1 - 60,000
    # validation size: 60,001 - 67,500
    # test size: 67,5001 - 75,000
    idx_train = range(0, 20000)
    idx_val = range(50000, 60000)
    idx_test = range(60000, 70000)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


# Confusion matrix
#    precision = tp / (tp + fp) = tp / total_predicted_positive
#    recall = tp / (tp + fn) = tp / total_actual_positive
#    f1 = 2 * (precision * recall) / (precision + recall)
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    acc = correct / len(labels)
    confusion_m = confusion_matrix(labels, preds).ravel()
    return (acc, confusion_m)

def accuracy_new(output, labels):
    print("output:", output)
    print("labels:", labels)
    preds = output.max(1)[1].type_as(labels)
    print("predictions:", preds)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    acc = correct / len(labels)
    print("correct:", correct, "total:", len(labels), "accuracy:", acc)
    return acc


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
