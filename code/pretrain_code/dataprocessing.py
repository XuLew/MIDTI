import torch
import numpy as np

import scipy.sparse as sp

# from torch_data import read_csv
# from pretrain_code.torch_data import read_csv
from code.pretrain_code.torch_data import read_csv


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + np.eye(adj.shape[0]))
    return torch.Tensor(adj_normalized)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj) 
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten() 
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0. 
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) 
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).toarray()

def normalize_features(feat):

    degree = np.asarray(feat.sum(1)).flatten()

    degree[degree == 0.] = np.inf
    degree_inv = 1./ degree
    degree_inv_mat = sp.diags([degree_inv], [0])
    feat_norm = degree_inv_mat.dot(feat)

    return torch.Tensor(feat_norm)



def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] != 0: 
                edge_index[0].append(i)
                edge_index[1].append(j)
    return torch.LongTensor(edge_index)


def data_pre(args):
    dataset = dict()
    if args.name == 'drug':
        "--drug sim--"
        "sim drug drug "
        dd_dr_matrix = read_csv(args.pretrain_dataset_path + 'Sim_mat_drug_drug.csv')
        dd_dr_matrix = preprocess_adj(dd_dr_matrix)
        dd_dr_edge_index = get_edge_index(dd_dr_matrix) 
        dataset['dd_dr'] = {'data_matrix': dd_dr_matrix, 'edges': dd_dr_edge_index}

        "sim drug disease "
        dd_di_matrix = read_csv(args.pretrain_dataset_path + 'Sim_mat_drug_disease.csv')
        dd_di_matrix = preprocess_adj(dd_di_matrix)
        dd_di_edge_index = get_edge_index(dd_di_matrix)
        dataset['dd_di'] = {'data_matrix': dd_di_matrix, 'edges': dd_di_edge_index}

        "sim drug se"
        dd_se_matrix = read_csv(args.pretrain_dataset_path + 'Sim_mat_drug_se.csv')
        dd_se_matrix = preprocess_adj(dd_se_matrix)
        dd_se_edge_index = get_edge_index(dd_se_matrix) 
        dataset['dd_se'] = {'data_matrix': dd_se_matrix, 'edges': dd_se_edge_index}

        "sim drug structure "
        dd_str_matrix = read_csv(args.pretrain_dataset_path + 'Sim_mat_drug_structure.csv')
        dd_str_matrix = preprocess_adj(dd_str_matrix)
        dd_str_edge_index = get_edge_index(dd_str_matrix)
        dataset['dd_str'] = {'data_matrix': dd_str_matrix, 'edges': dd_str_edge_index}

        "sim drug protein "
        dd_pro_matrix = read_csv(args.pretrain_dataset_path + 'Sim_mat_drug_protein.csv')
        dd_pro_matrix = preprocess_adj(dd_pro_matrix)
        dd_pro_edge_index = get_edge_index(dd_pro_matrix)
        dataset['dd_pro'] = {'data_matrix': dd_pro_matrix, 'edges': dd_pro_edge_index}
    else:
        "--protein sim--"
        "sim protein protein"
        pp_pro_matrix = read_csv(args.pretrain_dataset_path + 'Sim_mat_protein_protein.csv')
        pp_pro_matrix = preprocess_adj(pp_pro_matrix)
        pp_pro_edge_index = get_edge_index(pp_pro_matrix)  
        dataset['pp_pro'] = {'data_matrix': pp_pro_matrix, 'edges': pp_pro_edge_index}

        "sim protein disease"
        pp_di_matrix = read_csv(args.pretrain_dataset_path + 'Sim_mat_protein_disease.csv')
        pp_di_matrix = preprocess_adj(pp_di_matrix)
        pp_di_edge_index = get_edge_index(pp_di_matrix)
        dataset['pp_di'] = {'data_matrix': pp_di_matrix, 'edges': pp_di_edge_index}

        "sim protein sequence"
        pp_seq_matrix = read_csv(args.pretrain_dataset_path + 'Sim_mat_protein_sequence.csv')
        pp_seq_matrix = preprocess_adj(pp_seq_matrix)
        pp_seq_edge_index = get_edge_index(pp_seq_matrix)  
        dataset['pp_seq'] = {'data_matrix': pp_seq_matrix, 'edges': pp_seq_edge_index}

        "sim protein drug"
        pp_dr_matrix = read_csv(args.pretrain_dataset_path + 'Sim_mat_protein_drug.csv')
        pp_dr_matrix = preprocess_adj(pp_dr_matrix)
        pp_dr_edge_index = get_edge_index(pp_dr_matrix)  
        dataset['pp_dr'] = {'data_matrix': pp_dr_matrix, 'edges': pp_dr_edge_index}

    return dataset

