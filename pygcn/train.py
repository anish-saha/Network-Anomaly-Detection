from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import confusion_matrix

import torch
import torch.nn.functional as F
import torch.optim as optim

#from pygcn.utils import load_data, accuracy
from pygcn.models import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def load_data(path="data/twitter/", dataset="twitter"):
    """Load filtered Twitter dataset"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.features".format(path, dataset),
                                        delimiter=',',
                                        dtype=np.dtype(str))
    # shuffle the rows
    idx_features_labels = np.delete(idx_features_labels, np.s_[0], axis=0)
    np.random.shuffle(idx_features_labels)
    #print("idx_features_labels:", idx_features_labels)

    # Delete columns that are not required
    idx_features_labels = np.delete(idx_features_labels, np.s_[1:3], axis=1)
    #print("after removing columns idx_features_labels:", idx_features_labels)

    # Extract features
    features = sp.csr_matrix(idx_features_labels[:, 2:], dtype=np.float32)

    # Extract labels
    labels = encode_onehot(idx_features_labels[:, 1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    #print("idx:", idx)
    idx_map = {j: i for i, j in enumerate(idx)}
    #print("idx_map:", idx_map)

    edges_unordered = np.genfromtxt("{}{}.graph".format(path, dataset),
                                    delimiter=',',
                                    dtype=np.int32)
    edges_unordered = np.delete(edges_unordered, np.s_[0], axis=0)
    #print("edges_unordered:", edges_unordered)
    #print("edges_unordered.flatten():", edges_unordered.flatten())
    #print("edges_unordered.shape:", edges_unordered.shape)
    x = list(map(idx_map.get, edges_unordered.flatten()))
    #print("x:", x)
    y = [0 if i is None else i for i in x]
    #print("y:", y)
    edges = np.array(y, dtype=np.int32)
    edges = edges.reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    #print("features:", features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    #print("adj:", adj)

    # total size is 75624 entries
    # train size: 1 - 60,000
    # validation size: 60,001 - 67,500
    # test size: 67,5001 - 75,000
    idx_train = range(0, 20000)
    idx_val = range(50000, 60000)
    idx_test = range(60000, 70000)

    # small data-set
    #idx_train = range(0, 5)
    #idx_val = range(5, 8)
    #idx_test = range(5, 10)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data_cora(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('###Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    #print("idx_features_labels:", type(idx_features_labels), idx_features_labels.shape)
    #print("idx_features_labels:", idx_features_labels[0])
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    #print("features:", type(features), features.shape)
    #print("features:", features[0])
    labels = encode_onehot(idx_features_labels[:, -1])
    #print("labels:", type(labels), labels.shape)
    #print("labels:", labels[0])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

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

#
# Confusion matrix
#    precision = tp / (tp + fp) = tp / total_predicted_positive
#    recall = tp / (tp + fn) = tp / total_actual_positive
#    f1 = 2 * (precision * recall) / (precision + recall)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)




# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()
#print("idx_train:", idx_train)

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train, confusion_matrix = accuracy(output[idx_train], labels[idx_train])
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()

    print('###### Epoch: {:04d} ######'.format(epoch+1))
    print('Training: Loss={:04f} Accuracy={:04f}'.format(loss_train.item(),acc_train.item()))
    print('Training: Confusion Matrix(tn,fp,fn,tp)=', confusion_matrix)

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val, confusion_matrix = accuracy(output[idx_val], labels[idx_val])
    print('Validation: Loss={:04f} Accuracy={:04f}'.format(loss_val.item(),acc_val.item()))
    print('Validation: Confusion Matrix(tn,fp,fn,tp)=', confusion_matrix)

def test():
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test, confusion_matrix = accuracy(output[idx_test], labels[idx_test])
        print("==========Test set results==========")
        print("Loss= {:.4f}".format(loss_test.item()),
            "Accuracy= {:.4f}".format(acc_test.item()))
        print('Confusion Matrix (tn, fp, fn, tp)=', confusion_matrix)



# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
