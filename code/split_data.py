import numpy as np
import scipy.sparse as sp

def splitData(dataY, splitPath, nfold, seed_cross, crossKey='cross'):

    neg_pos_ratio = 1 #正负样本比例 1:1 1:5 1:10

    # 返回的是a=1对应的位置(二维数组)，需要用“行”(第0维)和“列”(第1维)表示
    index_pos = np.array(np.where(dataY == 1)) # (2,1923)
    index_neg = np.array(np.where(dataY == 0)) # (2,1068573)

    pos_num = len(index_pos[0])  # 1923
    neg_num = int(pos_num * neg_pos_ratio)  # 1923

    np.random.seed(seed_cross)
    np.random.shuffle(index_pos.T) # 多维矩阵中，shuffle只对第一维（行）做打乱顺序操作
    np.random.seed(seed_cross)
    np.random.shuffle(index_neg.T)
    index_neg = index_neg[:, : neg_num]  # (2,1923)

    # n折交叉检验，交叉检验的数据集cross_pos/neg下标对n求模 0,1,2,3,4
    cross_fold_index_pos = np.array([temp % nfold for temp in range(len(index_pos[0]))])
    cross_fold_index_neg = np.array([temp % nfold for temp in range(len(index_neg[0]))])

    #第一轮，模为0的做测试集，其他做训练集；第二轮，1为测试集...
    for kfold in range(nfold):
        cross_tra_fold_pos = index_pos.T[cross_fold_index_pos != kfold]
        cross_tes_fold_pos = index_pos.T[cross_fold_index_pos == kfold]
        cross_tra_fold_neg = index_neg.T[cross_fold_index_neg != kfold]
        cross_tes_fold_neg = index_neg.T[cross_fold_index_neg == kfold]

        cross_tra_fold = np.vstack((cross_tra_fold_pos, cross_tra_fold_neg))
        cross_tes_fold = np.vstack((cross_tes_fold_pos, cross_tes_fold_neg))

        cross_tra_data = np.hstack((cross_tra_fold, dataY[cross_tra_fold[:, 0], cross_tra_fold[:, 1]].reshape(-1, 1)))
        cross_tes_data = np.hstack((cross_tes_fold, dataY[cross_tes_fold[:, 0], cross_tes_fold[:, 1]].reshape(-1, 1)))
        cross_tra_matx = sp.coo_matrix((cross_tra_data[:, 2], (cross_tra_data[:, 0],cross_tra_data[:, 1])), shape=(dataY.shape[0],dataY.shape[1])).toarray()
        cross_tes_matx = sp.coo_matrix((cross_tes_data[:, 2], (cross_tes_data[:, 0],cross_tes_data[:, 1])), shape=(dataY.shape[0],dataY.shape[1])).toarray()


        np.savetxt(splitPath + crossKey + '_tra_kfold' + str(kfold) + '_seed' + str(seed_cross) + '.txt', cross_tra_data, fmt='%d', delimiter=',')
        np.savetxt(splitPath + crossKey + '_tes_kfold' + str(kfold) + '_seed' + str(seed_cross) + '.txt', cross_tes_data, fmt='%d', delimiter=',')

        # np.savetxt(splitPath + crossKey + '_tra_kfold' + str(kfold) + '_seed' + str(seed_cross) + '_mat.txt', cross_tra_matx, fmt='%d', delimiter=',')
        # np.savetxt(splitPath + crossKey + '_tes_kfold' + str(kfold) + '_seed' + str(seed_cross) + '_mat.txt', cross_tes_matx, fmt='%d', delimiter=',')


        cross_tra_data_total = cross_tra_data[cross_tra_data[:, -1]==1][:, :-1]
        cross_tes_data_total = cross_tes_data[cross_tes_data[:, -1]==1][:, :-1]
        cross_tra_data_total[:,1]+=dataY.shape[0]
        cross_tes_data_total[:,1]+=dataY.shape[0]
        np.savetxt(splitPath + crossKey + '_tra_kfold' + str(kfold) + '_seed' + str(seed_cross) + '_total.txt', cross_tra_data_total, fmt='%d', delimiter=' ')
        np.savetxt(splitPath + crossKey + '_tes_kfold' + str(kfold) + '_seed' + str(seed_cross) + '_total.txt', cross_tes_data_total, fmt='%d', delimiter=' ')

    return


if __name__ == "__main__":

    nfold = 5
    seed_cross = 1

    DATASET = 'LuoDTI'
    dataPath = '../dataset/' + DATASET + '/'
    usedDataPath = dataPath + 'used_data/'
    splitPath = dataPath + 'input/' #input5 #input10

    dataY = np.loadtxt(usedDataPath + 'mat_drug_protein.txt', dtype=float, delimiter=' ')
    splitData(dataY, splitPath, nfold, seed_cross)





