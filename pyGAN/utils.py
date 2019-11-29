import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def load_data(path="./data/twitter/", dataset="twitter"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.features_small".format(path, dataset),
                                        delimiter=',',
                                        dtype=np.dtype(str))

    # Delete rows and columns that are not required
    idx_features_labels = np.delete(idx_features_labels, np.s_[0], axis=0)
    idx_features_labels = np.delete(idx_features_labels, np.s_[1:3], axis=1)
    # print("after removing columns idx_features_labels:", idx_features_labels)

    # now the format is:
    #     node, label, feature1, feature2, ... featureN
    # shuffle the rows
    #idx_features_labels = idx_features_labels[idx_features_labels[:, 1].argsort()]

    idx_train = list(range(0, 5))
    idx_val = list(range(5, 8))
    idx_test = list(range(8, 11))

    #np.random.shuffle(idx_features_labels)
    #idx_train = list(range(0, 50000))
    #np.random.shuffle(idx_train)
    #idx_val = list(range(50000, 60000))
    #np.random.shuffle(idx_val)
    #idx_test = list(range(60000, 70000))
    #np.random.shuffle(idx_test)

    #idx_train = list(range(0, 5000)) + list(range(10000, 15000))
    #np.random.shuffle(idx_train)
    #idx_val = list(range(5000, 5500)) + list(range(15000, 15500))
    #np.random.shuffle(idx_val)
    #idx_test = list(range(6000, 6500)) + list(range(16000, 16500))
    #np.random.shuffle(idx_test)

    # Extract features
    features = sp.csr_matrix(idx_features_labels[:, 2:], dtype=np.float32)

    # Extract labels
    labels = encode_onehot(idx_features_labels[:, 1])
    # print("labels:", labels)

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    # Extract edges from the graph
    edges_unordered = np.genfromtxt("{}{}.graph_small".format(path, dataset),
                                    delimiter=',',
                                    dtype=np.int32)

    # Delete rows that we don't need
    edges_unordered = np.delete(edges_unordered, np.s_[0], axis=0)

    x = list(map(idx_map.get, edges_unordered.flatten()))
    y = [0 if i is None else i for i in x]
    edges = np.array(y, dtype=np.int32)
    edges = edges.reshape(edges_unordered.shape)
    print("edges_unordered shape:", edges_unordered.shape)
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

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def load_data_cora(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

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
    return correct / len(labels)

