# -*-coding:utf-8-*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np


class GraphConvolution(nn.Module):


    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, adj, input):
        try:
            input = input.float()
        except:
            pass

        support = torch.matmul(input, self.weight) #.clone()

        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output



'''Heterogeneous graph encoder'''

class GCN_hete(nn.Module):

    def __init__(self, hidden_c, output_hete):
        super(GCN_hete, self).__init__()

        self.hete = GraphConvolution(hidden_c, output_hete)
        self.hete2 = GraphConvolution(output_hete, output_hete)
        self.hete3 = GraphConvolution(output_hete, output_hete)


    def forward(self, adj_c, features_c):
        
        # gcn_layer=3
        out1 = torch.relu(self.hete(adj_c, features_c))
        out2 = torch.relu(self.hete2(adj_c, out1))
        out3 = torch.relu(self.hete3(adj_c, out2))
        return out1, out2, out3


'''Homogeneous graph encoder'''

class GCN_homo(nn.Module):

    def __init__(self, hidden_homo, output_homo):
        super(GCN_homo, self).__init__()

        self.gcn_homo = GraphConvolution(hidden_homo, output_homo)
        self.gcn_homo2 = GraphConvolution(output_homo, output_homo)
        self.gcn_homo3 = GraphConvolution(output_homo, output_homo)



    def forward(self, adj, features): 

        # gcn_layer=3
        out_d1 = torch.relu(self.gcn_homo(adj, features))
        out_d2 = torch.relu(self.gcn_homo2(adj, out_d1))
        out_d3 = torch.relu(self.gcn_homo3(adj, out_d2))
        return out_d1, out_d2, out_d3



'''Bipartite graph encoder''' 

class GCN_bi(nn.Module):

    def __init__(self, hidden_bi, output_bi):
        super(GCN_bi, self).__init__()

        # inter drug-protein graph
        self.gcn_bi_dp = GraphConvolution(hidden_bi, output_bi)
        self.gcn_bi_dp2 = GraphConvolution(output_bi, output_bi)
        self.gcn_bi_dp3 = GraphConvolution(output_bi, output_bi)


    def forward (self, adj_dp, features_dp) :

        # gcn_layer=3
        out_dp1 = torch.relu(self.gcn_bi_dp(adj_dp, features_dp))
        out_dp2 = torch.relu(self.gcn_bi_dp2(adj_dp, out_dp1))
        out_dp3 = torch.relu(self.gcn_bi_dp3(adj_dp, out_dp2))
        return out_dp1, out_dp2, out_dp3
        
