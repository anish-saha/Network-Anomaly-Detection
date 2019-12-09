import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, 16)
        self.gc2 = GraphConvolution(16, nhid)
        self.gc3 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x1 = F.leaky_relu(self.gc1(x, adj))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.leaky_relu(self.gc2(x1, adj))
        x2 = F.dropout(x2, self.dropout, training=self.training)
        x3 = self.gc3(x2, adj)
        return F.log_softmax(x3, dim=1)
