import math
from torch.nn.modules.activation import Sigmoid
from torch.nn.modules.pooling import AvgPool2d
from torch.nn.modules.conv import Conv1d
from torch.nn.parameter import Parameter
# from torch_geometric.nn.conv.gcn_conv import GCNConv
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.nn import GCNConv

# torch.backends.cudnn.enabled = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

    
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

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
        # size()函数主要是用来统计矩阵元素个数，或矩阵某一维上的元素个数的函数  size（1）为行
        stdv = 1. / math.sqrt(self.weight.size(1))  # sqrt() 方法返回数字x的平方根。
        # uniform()方法将随机生成下一个实数，它在 [x, y] 范围内
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, adj, input):
        try:
            input = input.float()
        except:
            pass
        support = torch.matmul(input, self.weight.clone())
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class ReGCNs(nn.Module):
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(ReGCNs, self).__init__()
        self.args = args

        self.gcn_x1_dr = GraphConvolution(self.args.drug_number, self.args.f)
        self.gcn_x2_dr = GraphConvolution(self.args.f, self.args.f)
        self.gcn_x1_di = GraphConvolution(self.args.drug_number, self.args.f)
        self.gcn_x2_di = GraphConvolution(self.args.f, self.args.f)
        self.gcn_x1_se = GraphConvolution(self.args.drug_number, self.args.f)
        self.gcn_x2_se = GraphConvolution(self.args.f, self.args.f)
        self.gcn_x1_str = GraphConvolution(self.args.drug_number, self.args.f)
        self.gcn_x2_str = GraphConvolution(self.args.f, self.args.f)
        self.gcn_x1_pro = GraphConvolution(self.args.drug_number, self.args.f)
        self.gcn_x2_pro = GraphConvolution(self.args.f, self.args.f)

        self.gcn_y1_pro = GraphConvolution(self.args.protein_number, self.args.f)
        self.gcn_y2_pro = GraphConvolution(self.args.f, self.args.f)
        self.gcn_y1_di = GraphConvolution(self.args.protein_number, self.args.f)
        self.gcn_y2_di = GraphConvolution(self.args.f, self.args.f)
        self.gcn_y1_seq = GraphConvolution(self.args.protein_number, self.args.f)
        self.gcn_y2_seq = GraphConvolution(self.args.f, self.args.f)
        self.gcn_y1_dr = GraphConvolution(self.args.protein_number, self.args.f)
        self.gcn_y2_dr = GraphConvolution(self.args.f, self.args.f)
        # self.gcn_x1_dr = GCNConv(self.args.drug_number, self.args.f)
        # self.gcn_x2_dr = GCNConv(self.args.f, self.args.f)
        # self.gcn_x1_di= GCNConv(self.args.drug_number, self.args.f)
        # self.gcn_x2_di = GCNConv(self.args.f, self.args.f)
        # self.gcn_x1_se = GCNConv(self.args.drug_number, self.args.f)
        # self.gcn_x2_se = GCNConv(self.args.f, self.args.f)
        # self.gcn_x1_str = GCNConv(self.args.drug_number, self.args.f)
        # self.gcn_x2_str = GCNConv(self.args.f, self.args.f)
        # self.gcn_x1_pro = GCNConv(self.args.drug_number, self.args.f)
        # self.gcn_x2_pro = GCNConv(self.args.f, self.args.f)
        #
        # self.gcn_y1_pro = GCNConv(self.args.protein_number, self.args.f)
        # self.gcn_y2_pro = GCNConv(self.args.f, self.args.f)
        # self.gcn_y1_di = GCNConv(self.args.protein_number, self.args.f)
        # self.gcn_y2_di = GCNConv(self.args.f, self.args.f)
        # self.gcn_y1_seq = GCNConv(self.args.protein_number, self.args.f)
        # self.gcn_y2_seq = GCNConv(self.args.f, self.args.f)
        # self.gcn_y1_dr = GCNConv(self.args.protein_number, self.args.f)
        # self.gcn_y2_dr = GCNConv(self.args.f, self.args.f)

        self.globalAvgPool_x = nn.AvgPool2d((self.args.f, self.args.drug_number), (1, 1)) #以前的，不再用了
        self.globalAvgPool_y = nn.AvgPool2d((self.args.f, self.args.protein_number), (1, 1))
        # self.globalAvgPool_x = AvgPool2d((self.args.drug_number, self.args.f), (1, 1))
        # self.globalAvgPool_y = AvgPool2d((self.args.protein_number, self.args.f), (1, 1))
        self.fc1_x = nn.Linear(in_features=self.args.view_d,
                             out_features=5*self.args.view_d)
        self.fc2_x = nn.Linear(in_features=5*self.args.view_d,
                             out_features=self.args.view_d)
        self.f_d = nn.Linear(in_features=self.args.f,
                               out_features=self.args.d_out_channels)

        self.fc1_y = nn.Linear(in_features=self.args.view_p,
                             out_features=5 * self.args.view_p)
        self.fc2_y = nn.Linear(in_features=5 * self.args.view_p,
                             out_features=self.args.view_p)
        self.f_p = nn.Linear(in_features=self.args.f,
                             out_features=self.args.p_out_channels)
        self.sigmoidx = Sigmoid()
        self.sigmoidy = Sigmoid()

        self.cnn_drug = Conv1d(in_channels=self.args.view_d,
                              out_channels=self.args.d_out_channels,
                              kernel_size=(self.args.f, 1),
                              stride=1,
                              bias=True)

        self.cnn_pro = Conv1d(in_channels=self.args.view_p,
                              out_channels=self.args.p_out_channels,
                              kernel_size=(self.args.f, 1),
                              stride=1,
                              bias=True)

    #在forward函数中，需要传入边的信息，也就是edge_index
    def forward(self, data):
        torch.manual_seed(1)
        if self.args.name == 'drug':
            x_d = torch.eye(self.args.drug_number) 
            # x_d = torch.randn(self.args.drug_number, self.args.f)  # 随机生成初始特征向量矩阵
            # gcn_conv输入节点特征矩阵x和邻接关系edge_index，还有一项edge_weight
            # 两层GCN
            # adj, input
            x_d_dr1 = torch.relu(self.gcn_x1_dr(data['dd_dr']['data_matrix'].to(device), x_d.to(device)))
            x_d_dr2 = torch.relu(self.gcn_x2_dr(data['dd_dr']['data_matrix'].to(device), x_d_dr1))
            #
            x_d_di1 = torch.relu(self.gcn_x1_di(data['dd_di']['data_matrix'].to(device), x_d.to(device)))
            x_d_di2 = torch.relu(self.gcn_x2_di(data['dd_di']['data_matrix'].to(device), x_d_di1))

            x_d_se1 = torch.relu(self.gcn_x1_se(data['dd_se']['data_matrix'].to(device), x_d.to(device)))
            x_d_se2 = torch.relu(self.gcn_x2_se(data['dd_se']['data_matrix'].to(device), x_d_se1))

            x_d_str1 = torch.relu(self.gcn_x1_str(data['dd_str']['data_matrix'].to(device), x_d.to(device)))
            x_d_str2 = torch.relu(self.gcn_x2_str(data['dd_str']['data_matrix'].to(device), x_d_str1))

            x_d_pro1 = torch.relu(self.gcn_x1_pro(data['dd_pro']['data_matrix'].to(device), x_d.to(device)))
            x_d_pro2 = torch.relu(self.gcn_x2_pro(data['dd_pro']['data_matrix'].to(device), x_d_pro1))

            # 多通道注意力

            XM = torch.cat((x_d_dr2, x_d_di2, x_d_se2, x_d_str2, x_d_pro2), 1).t()  # torch.cat((x, x, x), 1)：横向拼接
            # s个通道（ 相似性图数）
            XM = XM.view(1, self.args.view_d, self.args.f, -1)  # 1,4,512,708
            # print(XM.shape)

            x_channel_attenttion = self.globalAvgPool_x(XM) # 1,5,1,1
            # x_channel_attenttion = x_channel_attenttion.view(x_channel_attenttion.size(0), x_channel_attenttion.size(1))
            x_channel_attenttion = x_channel_attenttion.view(x_channel_attenttion.size(0),-1) #1,5
            x_channel_attenttion = self.fc1_x(x_channel_attenttion)
            x_channel_attenttion = torch.relu(x_channel_attenttion)
            x_channel_attenttion = self.fc2_x(x_channel_attenttion) #两个线性层
            x_channel_attenttion = self.sigmoidx(x_channel_attenttion)
            x_channel_attenttion = x_channel_attenttion.view(x_channel_attenttion.size(0), x_channel_attenttion.size(1), 1, 1) #1,5,1,1 #1,4,1,1
            XM_channel_attention = XM * x_channel_attenttion.expand_as(XM) # 1,5,708,512 # 1,4,708,512  #1,4,512,708
            #
            # XM_channel_attention = torch.relu(XM_channel_attention)
            # x_d_fe = torch.sum(XM_channel_attention, dim=(0,1)) #708,512

            x_d_fe = self.cnn_drug(XM_channel_attention)  # 1,4,512,708 → 1,708,1,708
            x_d_fe = x_d_fe.view(self.args.d_out_channels, self.args.drug_number).t()  # 1,708,1,708 → 708(out_dim),708(drug-number)  再转置

            # #不用多视图注意力
            # XM = torch.stack((x_d_dr2, x_d_di2, x_d_se2, x_d_str2, x_d_pro2),
            #                  0)  # torch.stack()的作用是在新的维度上把n个同size的矩阵进行拼接，dim=0即该函数会增加维度0，拼接后第0维的size为3 #5,708,512
            # # XM = XM.unsqueeze(0)  # 形状变为1*5*708*512,在最前面增加一个维度
            # x_d_fe = XM.mean(dim=0)  # 在0维求平均，然后第1维消失 5*708*512 708*512
            # x_d_fe = self.f_d(x_d_fe) #线性层变换维度，以708作为药物特征向量输出
            # print(x_d_fe.shape)
            ##

            x_d = x_d_fe.mm(x_d_fe.t())  # 重构相似性矩阵

            return x_d_fe, x_d


        elif self.args.name == 'protein':
            y_p = torch.eye(self.args.protein_number)
            # y_p = torch.randn(self.args.protein_number, self.args.f)  # 随机生成初始特征向量矩阵
            # print('y_p0:',y_p)
            # 两层GCN
            y_p_pro1 = torch.relu(self.gcn_y1_pro(data['pp_pro']['data_matrix'].to(device), y_p.to(device)))
            y_p_pro2 = torch.relu(self.gcn_y2_pro(data['pp_pro']['data_matrix'].to(device), y_p_pro1))

            y_p_di1 = torch.relu(self.gcn_y1_di(data['pp_di']['data_matrix'].to(device), y_p.to(device)))
            y_p_di2 = torch.relu(self.gcn_y2_di(data['pp_di']['data_matrix'].to(device), y_p_di1))

            y_p_seq1 = torch.relu(self.gcn_y1_seq(data['pp_seq']['data_matrix'].to(device), y_p.to(device)))
            y_p_seq2 = torch.relu(self.gcn_y2_seq(data['pp_seq']['data_matrix'].to(device), y_p_seq1))

            y_p_dr1 = torch.relu(self.gcn_y1_dr(data['pp_dr']['data_matrix'].to(device), y_p.to(device)))
            y_p_dr2 = torch.relu(self.gcn_y2_dr(data['pp_dr']['data_matrix'].to(device), y_p_dr1))


            # 多通道注意力

            YM = torch.cat((y_p_pro2, y_p_di2, y_p_seq2, y_p_dr2), 1).t()  # torch.cat((x, x, x), 1)：横向拼接
            # s个通道（ 相似性图数）
            YM = YM.view(1, self.args.view_p, self.args.f, -1)  # 1,4,512,1512
            # print(YM.shape)
            #
            y_channel_attenttion = self.globalAvgPool_y(YM)  # 1,4,1,1
            # y_channel_attenttion = y_channel_attenttion.view(y_channel_attenttion.size(0), y_channel_attenttion.size(1))
            y_channel_attenttion = y_channel_attenttion.view(y_channel_attenttion.size(0), -1)
            y_channel_attenttion = self.fc1_y(y_channel_attenttion)
            y_channel_attenttion = torch.relu(y_channel_attenttion)
            y_channel_attenttion = self.fc2_y(y_channel_attenttion)
            y_channel_attenttion = self.sigmoidy(y_channel_attenttion)
            y_channel_attenttion = y_channel_attenttion.view(y_channel_attenttion.size(0), y_channel_attenttion.size(1),1, 1)
            YM_channel_attention = YM * y_channel_attenttion.expand_as(YM)  # 1,4,1512,512 # 1,3,1512,512 #1,3,512,1512

            y_p_fe = self.cnn_pro(YM_channel_attention)  # 1,4,1512,1512  → 1,1512,1,1512
            y_p_fe = y_p_fe.view(self.args.p_out_channels, self.args.protein_number).t()  # 1,1512,1,1512 → 1512,1512(pro_number) → 转置

            # # 不用多视图注意力
            # YM = torch.stack((y_p_pro2, y_p_di2, y_p_seq2, y_p_dr2), 0)  # torch.stack()的作用是在新的维度上把n个同size的矩阵进行拼接，dim=0即该函数会增加维度0，拼接后第0维的size为3 #4,1512,512
            # y_p_fe= YM.mean(dim=0) # 在0维求平均，然后第1维消失 4*1512*512 1512*512
            # y_p_fe = self.f_p(y_p_fe)  # 线性层变换维度，以708作为药物特征向量输出
            # # print(y_p_fe.shape)
            # ##

            y_p = torch.mm(y_p_fe, y_p_fe.t())# 重构相似性矩阵

            return y_p_fe, y_p










