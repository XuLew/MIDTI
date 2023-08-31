# -*- coding: utf-8 -*-
"""
convert data into similarity between drug or protein network
using jaccard distance 

"""
import csv
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


def outputCSVfile(filename, data):
    csvfile = open(filename, 'w', newline="")
    writer = csv.writer(csvfile)
    writer.writerows(data)  # 写入多行数据
    csvfile.close()

# def transSim(dataS):
#     # dataS[1500:,:].fill(0) #令从1500行后面的每一行值为0
#     # dataS[:,1500:].fill(0) #令从1500列后面的每一列值为0
#     # 返回的是a>0对应的位置(二维数组)，需要用“行”(第0维)和“列”(第1维)表示
#     index_pos = np.array(np.where(dataS > 0))
#     index_neg = np.array(np.where(dataS == 0))
#
#     # index_pos.T 数据集中的位置坐标（行，列）
#     # Sim_data（行，列，关系矩阵中的值0/1） 包含正负样本 shape=(n, 3)
#     Sim_data = np.hstack((index_pos.T, dataS[index_pos[0], index_pos[1]].reshape(-1, 1)))
#     return Sim_data
#
# M_p=pd.read_table('mat_drug_protein.txt',sep=' ',header=None)
# M_p=M_p.values.T
# np.savetxt('mat_protein_drug.txt', M_p, fmt='%d', delimiter=' ')


DATASET = 'LuoDTI'
Nets = ['drug_drug', 'drug_disease', 'drug_se', 'drug_protein', 'protein_protein', 'protein_disease', 'protein_drug']

for net in Nets:
    # inputID=usedDataPath+'mat_'+net+'.txt'
    inputID = 'mat_' + net + '.txt'
    M=pd.read_table(inputID,sep=' ', header=None)
    # jaccard distance
    Sim=1-pdist(M,'jaccard')
    Sim=squareform(Sim)
    # Sim=Sim+np.eye(len(Sim)) # 加上自环，对角线相似性为1
    Sim=np.nan_to_num(Sim)

    # #output csv file
    # 如有需要参考DTINet-with-python-master项目的compute similarity.py和utils.py文件
    outputID='Sim_mat_'+net+'.csv'
    outputCSVfile(outputID, Sim)

    # # output txt file
    # outputID = usedDataPath+'Sim_mat_' + net + '.txt'
    # np.savetxt(outputID, Sim, fmt='%.10f', delimiter=',') #%d表示整数，%.10f为保留小数点后10位

    # Sim=Sim-np.eye(len(Sim)) #不含对角线的
    # Sim_data = transSim(Sim)
    # # np.savetxt(usedDataPath + 'Sim_' + net + '_row_column_value' + '.txt', Sim_data,
    # #            fmt='%d %d %.10f', delimiter=' ')
    # np.savetxt('Sim_' + net + '_row_column_value' + '.txt', Sim_data,
    #            fmt='%d %d %.10f', delimiter=' ')

# #write chemical similarity to networks
# M_d=pd.read_table(usedDataPath+'Similarity_Matrix_Drugs.txt',sep='    ',header=None)
M_d = pd.read_table('Similarity_Matrix_Drugs.txt', sep='    ', header=None)

M_d=M_d-np.eye(len(M_d)) #不含对角线的
M_d = M_d.values

outputCSVfile('Sim_mat_drug_structure.csv', M_d)
# np.savetxt(usedDataPath+'Sim_mat_drug_structure.txt', M_d, fmt='%.10f', delimiter = ',')


# Sim_data_d = transSim(M_d)
# np.savetxt(usedDataPath + 'Sim_drug_structure_row_column_value' + '.txt', Sim_data_d,
#                fmt='%d %d %.10f', delimiter=' ')
# np.savetxt('Sim_drug_structure_row_column_value' + '.txt', Sim_data_d,
#                fmt='%d %d %.10f', delimiter=' ')

#write sequence similarity to networks
M_p = pd.read_table('Similarity_Matrix_Proteins.txt', sep=' ', header=None)
# M_p=pd.read_table('Similarity_Matrix_Proteins.txt',sep=' ',header=None)
M_p=M_p/100
M_p=M_p-np.eye(len(M_p)) #不含对角线的
M_p=M_p.values
outputCSVfile('Sim_mat_protein_sequence.csv', M_p)
# np.savetxt(usedDataPath+'Sim_mat_protein_sequence.txt', M_p, fmt='%.10f', delimiter = ',')


# Sim_data_p = transSim(M_p)
# np.savetxt(usedDataPath + 'Sim_protein_sequence_row_column_value' + '.txt', Sim_data_p,
#                fmt='%d %d %.10f', delimiter=' ')
# np.savetxt('Sim_protein_sequence_row_column_value' + '.txt', Sim_data_p,
#                fmt='%d %d %.10f', delimiter=' ')



# M_p=pd.read_table('drug_matrix.txt',sep=' ',header=None)
# M_p=M_p.values
# outputCSVfile('drug_matrix.csv', M_p)
