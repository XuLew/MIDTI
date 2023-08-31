import torch
import torch.nn as nn

import torch.nn.functional as F
import torch
import math


from GCNLayer import GCN_homo, GCN_bi, GCN_hete


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')


class MHAtt(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super(MHAtt, self).__init__()

        self.linear_v = nn.Linear(hid_dim, hid_dim)
        self.linear_k = nn.Linear(hid_dim, hid_dim)
        self.linear_q = nn.Linear(hid_dim, hid_dim)
        self.linear_merge = nn.Linear(hid_dim, hid_dim)
        self.hid_dim = hid_dim
        self.dropout = dropout
        self.nhead = n_heads

        self.dropout = nn.Dropout(dropout)
        self.hidden_size_head = int(self.hid_dim / self.nhead)
    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.nhead,
            self.hidden_size_head
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.nhead,
            self.hidden_size_head
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.nhead,
            self.hidden_size_head
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask) #1,8,1,64
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.hid_dim
        ) #1,1,512

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1) #64

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k) #1,8,1,1

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


class DPA(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super(DPA, self).__init__()

        self.mhatt1 = MHAtt(hid_dim, n_heads, dropout)
        # self.mhatt1 = MultiAttn(hid_dim, n_heads)

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hid_dim)



    def forward(self, x, y, y_mask=None):

        # x as V while y as Q and K
        # x = self.norm1(x + self.dropout1(
        #     self.mhatt1(x, x, y, y_mask)
        # ))
        x = self.norm1(x+self.dropout1(
            self.mhatt1(y, y, x, y_mask)
        ))
        # x = self.norm1(x + self.dropout1(
        #     self.mhatt1(x, y, y_mask, y_mask)
        # ))

        return x


class SA(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super(SA, self).__init__()

        self.mhatt1 = MHAtt(hid_dim, n_heads, dropout)
        # self.mhatt1 = MultiAttn(hid_dim, n_heads)

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hid_dim)



    def forward(self, x, mask=None):

        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, mask)
        ))
        # x = self.norm1(x + self.dropout1(
        #     self.mhatt1(x, x, mask, mask)
        # ))

        return x


class encoder_cross_att(nn.Module):
    def __init__(self, dim, nhead, dropout, layers):
        super(encoder_cross_att, self).__init__()
        # self.encoder_layers = nn.ModuleList([SEA(dim, dropout) for _ in range(layers)])
        self.encoder_layers = nn.ModuleList([SA(dim, nhead, dropout) for _ in range(layers)])
        self.decoder_sa = nn.ModuleList([SA(dim, nhead, dropout) for _ in range(layers)])
        self.decoder_coa = nn.ModuleList([DPA(dim, nhead, dropout)  for _ in range(layers)])
        self.layer_coa = layers
    def forward(self, drug_vector, protein_vector):
        for i in range(self.layer_coa):
            drug_vector = self.encoder_layers[i](drug_vector, None)  # self-attention
        for i in range(self.layer_coa):
            protein_vector = self.decoder_sa[i](protein_vector, None)
            protein_vector = self.decoder_coa[i](protein_vector, drug_vector, None)# co-attention

        return drug_vector, protein_vector


class stack_cross_att(nn.Module):
    def __init__(self, dim, nhead, dropout):
        super(stack_cross_att, self).__init__()
        self.sda = SA(dim, nhead, dropout)
        self.spa = SA(dim, nhead, dropout)
        self.coa_dp = DPA(dim, nhead, dropout)

    def forward(self, drug_vector,protein_vector ):
        drug_vector = self.sda(drug_vector, None)  # self-attention
        protein_vector = self.spa(protein_vector, None)  # self-attention
        protein_covector = self.coa_dp(protein_vector, drug_vector, None)  # co-attention

        return drug_vector, protein_covector


class inter_cross_att(nn.Module):
    def __init__(self, dim, nhead, dropout):
        super(inter_cross_att, self).__init__()
        self.sca = SA(dim, nhead, dropout)
        self.spa = SA(dim, nhead, dropout)
        self.coa_pc = DPA(dim, nhead, dropout)
        self.coa_cp = DPA(dim, nhead, dropout)

    def forward(self, drug_vector, protein_vector,):
        drug_vector = self.sca(drug_vector, None)  # self-attention
        protein_vector = self.spa(protein_vector, None)  # self-attention
        drug_covector = self.coa_pc(drug_vector, protein_vector, None)  # co-attention
        protein_covector = self.coa_cp(protein_vector, drug_vector, None)  # co-attention

        return drug_covector, protein_covector



class GCNLayer(nn.Module):
    def __init__(self, input_d, input_p, dim):
        super(GCNLayer, self).__init__()
        self.input_d = input_d
        self.input_p = input_p
        # self.lin_d = torch.nn.Linear(dim, dim)
        # self.lin_p = torch.nn.Linear(dim, dim)


        self.gcn_homo_d = GCN_homo(input_d, dim)
        self.gcn_homo_p = GCN_homo(input_p, dim)
        self.gcn_bi = GCN_bi(input_d+input_p, dim)
        self.gcn_hete = GCN_hete(input_d+input_p, dim)

    def forward(self, datasetF):
        # # 1层
        # x_d_dr1= self.gcn_intra_d(datasetF['dd']['matrix'].to(device), datasetF['dd']['feature'].to(device))
        # y_p_pro1= self.gcn_intra_p(datasetF['pp']['matrix'].to(device),
        #                                         datasetF['pp']['feature'].to(device))
        # dp1= self.gcn_inter(datasetF['dp']['matrix'].to(device), datasetF['dp']['feature'].to(device))
        # ddpp1= self.gcn_complex(datasetF['ddpp']['matrix'].to(device), datasetF['ddpp']['feature'].to(device))
        #
        # x_d_dr2 = dp1[:self.input_d, :]
        # y_p_pro2 = dp1[self.input_d:, :]
        #
        # x_d_dr3 = ddpp1[:self.input_d, :]
        # y_p_pro3 = ddpp1[self.input_d:, :]
        #
        # x_d_dr = torch.stack((x_d_dr1, x_d_dr2, x_d_dr3),0)  # torch.stack()的作用是在新的维度0上把n个同size的矩阵进行拼接.shape为3，n,512
        # y_p_pro = torch.stack((y_p_pro1, y_p_pro2, y_p_pro3), 0)

#         # 2层
#         x_d_dr1, x_d_dr2 = self.gcn_intra_d(datasetF['dd']['matrix'].to(device), datasetF['dd']['feature'].to(device))
#         y_p_pro1, y_p_pro2 = self.gcn_intra_p(datasetF['pp']['matrix'].to(device), datasetF['pp']['feature'].to(device))
#         dp1, dp2 = self.gcn_inter(datasetF['dp']['matrix'].to(device), datasetF['dp']['feature'].to(device))
#         ddpp1, ddpp2 = self.gcn_complex(datasetF['ddpp']['matrix'].to(device), datasetF['ddpp']['feature'].to(device))

#         x_d_dr3 = dp1[:self.input_d,:]
#         y_p_pro3 = dp1[self.input_d:, :]
#         x_d_dr4 = dp2[:self.input_d, :]
#         y_p_pro4 = dp2[self.input_d:, :]

#         x_d_dr5 = ddpp1[:self.input_d, :]
#         y_p_pro5 = ddpp1[self.input_d:, :]
#         x_d_dr6 = ddpp2[:self.input_d, :]
#         y_p_pro6 = ddpp2[self.input_d:, :]

#         x_d_dr = torch.stack((x_d_dr1, x_d_dr2,x_d_dr3, x_d_dr4,x_d_dr5, x_d_dr6), 0)  # torch.stack()的作用是在新的维度0上把n个同size的矩阵进行拼接.shape为6，n,512
#         y_p_pro = torch.stack((y_p_pro1, y_p_pro2, y_p_pro3, y_p_pro4, y_p_pro5, y_p_pro6), 0)
#         # print('x_d_dr.shape,y_p_pro.shape',x_d_dr.shape,y_p_pro.shape)

        # 3层
        x_d_dr1, x_d_dr2, x_d_dr3 = self.gcn_homo_d(datasetF['dd']['matrix'].to(device), datasetF['dd']['feature'].to(device))
        y_p_pro1, y_p_pro2, y_p_pro3 = self.gcn_homo_p(datasetF['pp']['matrix'].to(device), datasetF['pp']['feature'].to(device))
        dp1, dp2, dp3= self.gcn_bi(datasetF['dp']['matrix'].to(device), datasetF['dp']['feature'].to(device))
        ddpp1, ddpp2, ddpp3 = self.gcn_hete(datasetF['ddpp']['matrix'].to(device), datasetF['ddpp']['feature'].to(device))
        
        x_d_dr4 = dp1[:self.input_d,:]
        y_p_pro4 = dp1[self.input_d:, :]
        x_d_dr5 = dp2[:self.input_d, :]
        y_p_pro5 = dp2[self.input_d:, :]
        x_d_dr6 = dp3[:self.input_d, :]
        y_p_pro6 = dp3[self.input_d:, :]
        
        x_d_dr7 = ddpp1[:self.input_d, :]
        y_p_pro7 = ddpp1[self.input_d:, :]
        x_d_dr8 = ddpp2[:self.input_d, :]
        y_p_pro8 = ddpp2[self.input_d:, :]
        x_d_dr9 = ddpp3[:self.input_d, :]
        y_p_pro9 = ddpp3[self.input_d:, :]
        
        x_d_dr = torch.stack((x_d_dr1, x_d_dr2, x_d_dr3, x_d_dr4, x_d_dr5, x_d_dr6, x_d_dr7,x_d_dr8, x_d_dr9), 0)  # torch.stack()的作用是在新的维度0上把n个同size的矩阵进行拼接.shape为9，n,512
        y_p_pro = torch.stack((y_p_pro1, y_p_pro2, y_p_pro3, y_p_pro4, y_p_pro5, y_p_pro6, y_p_pro7, y_p_pro8, y_p_pro9), 0)
        # print('x_d_dr.shape,y_p_pro.shape',x_d_dr.shape,y_p_pro.shape)

        # # 4层
        # x_d_dr1, x_d_dr2, x_d_dr3, x_d_dr4= self.gcn_intra_d(datasetF['dd']['matrix'].to(device), datasetF['dd']['feature'].to(device))
        # y_p_pro1, y_p_pro2, y_p_pro3, y_p_pro4= self.gcn_intra_p(datasetF['pp']['matrix'].to(device), datasetF['pp']['feature'].to(device))
        # dp1, dp2, dp3, dp4= self.gcn_inter(datasetF['dp']['matrix'].to(device), datasetF['dp']['feature'].to(device))
        # ddpp1, ddpp2, ddpp3, ddpp4 = self.gcn_complex(datasetF['ddpp']['matrix'].to(device), datasetF['ddpp']['feature'].to(device))
        #
        # x_d_dr5 = dp1[:self.input_d,:]
        # y_p_pro5 = dp1[self.input_d:, :]
        # x_d_dr6 = dp2[:self.input_d, :]
        # y_p_pro6 = dp2[self.input_d:, :]
        # x_d_dr7 = dp3[:self.input_d, :]
        # y_p_pro7 = dp3[self.input_d:, :]
        # x_d_dr8 = dp4[:self.input_d, :]
        # y_p_pro8 = dp4[self.input_d:, :]
        #
        # x_d_dr9 = ddpp1[:self.input_d, :]
        # y_p_pro9 = ddpp1[self.input_d:, :]
        # x_d_dr10 = ddpp2[:self.input_d, :]
        # y_p_pro10 = ddpp2[self.input_d:, :]
        # x_d_dr11 = ddpp3[:self.input_d, :]
        # y_p_pro11 = ddpp3[self.input_d:, :]
        # x_d_dr12 = ddpp4[:self.input_d, :]
        # y_p_pro12 = ddpp4[self.input_d:, :]
        #
        # x_d_dr = torch.stack((x_d_dr1, x_d_dr2,x_d_dr3, x_d_dr4, x_d_dr5, x_d_dr6, x_d_dr7, x_d_dr8, x_d_dr9, x_d_dr10,x_d_dr11, x_d_dr12), 0)  # torch.stack()的作用是在新的维度0上把n个同size的矩阵进行拼接.shape为12，n,512
        # y_p_pro = torch.stack((y_p_pro1, y_p_pro2, y_p_pro3, y_p_pro4, y_p_pro5, y_p_pro6, y_p_pro7, y_p_pro8, y_p_pro9, y_p_pro10, y_p_pro11, y_p_pro12), 0)
        # # print('x_d_dr.shape,y_p_pro.shape',x_d_dr.shape,y_p_pro.shape)

        x_d_dr = torch.transpose(x_d_dr, 0, 1)  # shape=(708,9,512)
        y_p_pro = torch.transpose(y_p_pro, 0, 1)  # shape=(1512,9,512)

        # print('x_d_dr, y_p_pro:', x_d_dr, y_p_pro)
        return x_d_dr, y_p_pro


class Dtis(nn.Module):
    def __init__(self, input_d, input_p, dim, layer_output, layer_IA, nhead, dropout, attention):
        super(Dtis, self).__init__()

        self.gcnlayer = GCNLayer(input_d, input_p, dim)
        self.co_attention = attention

        # the number of attention layers
        self.layer_IA = layer_IA

        # self.encoder_IA_ModuleList = encoder_cross_att(dim, nhead, dropout, layer_IA)
        # self.stack_IA_ModuleList = nn.ModuleList([stack_cross_att(dim, nhead, dropout) for _ in range(layer_IA)])

        self.IA_ModuleList = nn.ModuleList([inter_cross_att(dim, nhead, dropout) for _ in range(layer_IA)])
        self.dr_lin = nn.Linear(layer_IA*dim, dim)
        self.pro_lin = nn.Linear(layer_IA*dim, dim)
        # the number of output layers
        self.layer_output = layer_output

        # self.W_out = nn.ModuleList([nn.Linear(2*dim, dim),nn.Linear(dim, 128),nn.Linear(128, 64)]) #mlp=4
        self.W_out = nn.ModuleList([nn.Linear(2*dim, dim), nn.Linear(dim, 128)]) #mlp=3
        self.W_interaction = nn.Linear(128, 2) #mlp=3层，前面两层隐藏层

        # self._init_weight()


    def forward(self, train_dataset, datasetF):

        global drug_vector_co, protein_vector_co, drug_vector, protein_vector
        x_d_dr, y_p_pro = self.gcnlayer(datasetF) # GCN模块
        # print('x_d_dr, y_p_pro:', x_d_dr, y_p_pro)



        # print(dataset)
        id0 = train_dataset[:,0].type(torch.long).to(device) # 索引要为long, byte 或 bool类型
        id1 = train_dataset[:,1].type(torch.long).to(device)
        interaction = train_dataset[:,2].type(torch.long).to(device)

        drugs = x_d_dr[id0, :, :]  # (1,9,512)
        proteins = y_p_pro[id1, :, :]   # (1,9,512)

        # #encoder
        # drug_vector, protein_vector = self.encoder_IA_ModuleList(drugs, proteins)
        # #stack
        # for i in range(self.layer_IA):
        #     drug_vector, protein_vector = self.stack_IA_ModuleList[i](drugs, proteins)

        # IA(inteactive attention)
        for i in range(self.layer_IA): #目标模型，将co-attention每层的输出拼接在一起
            drug_vector, protein_vector = self.IA_ModuleList[i](drugs, proteins) #1,6,512
            if i ==0:
                drug_vector_co, protein_vector_co = drug_vector, protein_vector
            else:
                drug_vector_co, protein_vector_co = torch.cat([drug_vector_co, drug_vector], dim=-1), torch.cat([protein_vector_co, protein_vector], dim=-1)
                # print(drug_vector_co.shape, protein_vector_co.shape) #(1,9,512*3)
        drug_vector, protein_vector = self.dr_lin(drug_vector_co), self.pro_lin(protein_vector_co)

        drug_covector = drug_vector.mean(dim=1)
        protein_covector = protein_vector.mean(dim=1)
        
        # # 无共同注意力
        # drug_covector = drugs.mean(dim=1)
        # protein_covector = proteins.mean(dim=1)
        # ##
        

        """Concatenate the above two vectors and output the interaction."""
        # catenate the two vectors
        cat_vector = torch.cat((drug_covector, protein_covector), 1)
        cat_vector0 = cat_vector
        
        # cat_vector = torch.cat((drug_covector, protein_covector), 1)
        for j in range(self.layer_output-1):
            cat_vector = torch.tanh(self.W_out[j](cat_vector))
        predicted = self.W_interaction(cat_vector)
        # print('predicted_interaction_shape', predicted_interaction.shape)
        # print('predicted_interaction',predicted_interaction)
        # if i == 0:
        #     predicted_logits = predicted
        #     correct = interaction

        return cat_vector0, predicted, interaction


