from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import scipy.sparse as sp

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data, accuracy
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

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

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
        print("====================Test set results====================")
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
