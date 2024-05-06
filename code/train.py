import pickle
import timeit

import numpy as np
from sklearn.metrics.classification import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.metrics.ranking import roc_auc_score, roc_curve, average_precision_score
# from sklearn.metrics import matthews_corrcoef, f1_score
# from torch import optim, softmax

from torch.functional import Tensor
from torch.nn.functional import softmax
# from torch.nn import CrossEntropyLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.nn.utils.clip_grad import clip_grad_norm_


import random
import logging

import os
from collections import defaultdict
from torch.optim import Optimizer, SGD
import torch

from MIDTI import DTI_pre
from code.pretrain_code.dataprocessing import preprocess_adj, normalize_features
from code.pretrain_code.torch_data import read_txt, read_csv


def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] != 0: 
                edge_index[0].append(i)
                edge_index[1].append(j)
    return Tensor(edge_index)


class Lookahead(Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0

    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)

    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss

    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def add_param_group(self, param_group):
        param_group["counter"] = 0
        self.optimizer.add_param_group(param_group)

def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy')]


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def random_shuffle(dataset, seed):
    random.seed(seed)  # 2021 2345 1234
    random.shuffle(dataset)
    return dataset


class Trainer(object):
    def __init__(self, model):

        self.model = model

        self.optimizer_inner = SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.optimizer = Lookahead(self.optimizer_inner, k=0, alpha=0.5)


    def train(self, train_dataset, datasetF):  # , datasetF

        N = len(train_dataset)

        train_labels = []
        train_preds = []

        # self.optimizer.zero_grad()


        cat_vector, predicted, interaction = self.model(train_dataset, datasetF)

        criterion = CrossEntropyLoss().to(device)  # 交叉熵损失(Cross Entropy Loss, CE)，已经含有softmax
        loss = criterion(predicted, interaction)


        loss.backward()
        clip_grad_norm_(parameters=self.model.parameters(),
                        max_norm=5)
        self.optimizer.step()
        self.optimizer.zero_grad()

        preds = predicted.max(1)[1]
        # print('preds', preds)

        train_labels.extend(interaction.cpu().detach().numpy())  # T
        train_preds.extend(preds.cpu().detach().numpy())  # Y
        # print(train_labels, train_preds)

        train_acc = accuracy_score(train_labels, train_preds)

        return loss.item(), train_acc


class Tester(object):
    def __init__(self, model):
        self.model = model.to(device)
        # self.batch_size = batch_size

    def dev(self, epoch, dev_dataset, datasetF):
        test_labels = []
        test_preds = []
        test_scores = []

        # self.model.eval()
        N = len(dev_dataset)
        # print(N)

        with torch.no_grad():
            # _, _,
            cat_vector, predicted, interaction = self.model(dev_dataset, datasetF)
            print(cat_vector.shape, interaction.shape)

            cat_vector = cat_vector.cpu().detach().numpy()
            interaction = interaction.cpu().detach().numpy()

            ys = softmax(predicted, 1).cpu().detach().numpy()

            predicted_labels = list(map(lambda x: np.argmax(x),ys))
            predicted_scores = list(map(lambda x: x[1], ys))

            test_labels.extend(interaction)  # T
            test_preds.extend(predicted_labels)  # Y
            test_scores.extend(predicted_scores)  # S

        fpr, tpr, thresholds = roc_curve(test_labels, test_scores)  # T,S

        test_acc = accuracy_score(test_labels, test_preds)  # T,Y
        test_auc = roc_auc_score(test_labels, test_scores)  # T,S
        test_aupr = average_precision_score(test_labels, test_scores)
        precision = precision_score(test_labels, test_preds)  # T,Y
        recall = recall_score(test_labels, test_preds)  # T,Y

        f1 = f1_score(test_labels, test_preds)
        MCC = matthews_corrcoef(test_labels, test_preds)


        return test_labels, test_scores, test_acc, test_auc, test_aupr, precision, recall, f1, MCC

    def test(self, epoch, test_dataset, datasetF):
        test_labels = []
        test_preds = []
        test_scores = []

        N = len(test_dataset)
        # print(N)

        with torch.no_grad():

            cat_vector, predicted, interaction = self.model(test_dataset, datasetF)
            print(cat_vector.shape, interaction.shape)

            cat_vector = cat_vector.cpu().detach().numpy()
            interaction = interaction.cpu().detach().numpy()

            ys = softmax(predicted, 1).cpu().detach().numpy()

            predicted_labels = list(map(lambda x: np.argmax(x),ys))
            predicted_scores = list(map(lambda x: x[1], ys))

            test_labels.extend(interaction)  # T
            test_preds.extend(predicted_labels)  # Y
            test_scores.extend(predicted_scores)  # S
            # print('case_score:', test_scores)

        return test_labels, test_scores

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a+') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)


def train(DATASET, fold, save_auc, attention, random_seed, log_write=True):

    global best_train_acc, best_acc, best_epoch, best_aupr, best_loss_train, best_precision, best_f1, best_mcc, best_recall
    dir_input = ('../dataset/' + DATASET)

    
    train_dataset = np.loadtxt(dir_input + '/input/cross_tra_kfold{}_seed1.txt'.format(fold), dtype=int, delimiter=',')
    dev_dataset = np.loadtxt(dir_input + '/input/cross_tes_kfold{}_seed1.txt'.format(fold), dtype=int, delimiter=',')
    test_dataset = dev_dataset

    drug_protein_matrix = read_txt(dir_input + '/input/mat_drug_protein.txt', ' ')
    drug_num = len(drug_protein_matrix)
    protein_num = len(drug_protein_matrix[0])

    datasetF = dict()
    drug_sim_matrix = read_csv(dir_input + '/input/dim/drug_f512_matrix708.csv')
    drug_sim_matrix1 = preprocess_adj(drug_sim_matrix)
    drug_sim_edge_index = get_edge_index(drug_sim_matrix1)

    drug_feature = read_csv(dir_input + '/input/dim/drug_f512_feature708.csv')
    drug_feature = normalize_features(drug_feature)

    datasetF['dd'] = {'matrix': drug_sim_matrix1, 'edges': drug_sim_edge_index, 'feature': drug_feature}

    protein_sim_matrix = read_csv(dir_input + '/input/dim/protein_f512_matrix1512.csv')
    protein_sim_matrix1 = preprocess_adj(protein_sim_matrix)
    protein_sim_edge_index = get_edge_index(protein_sim_matrix1)

    protein_feature = read_csv(dir_input + '/input/dim/protein_f512_feature1512.csv')
    protein_feature = normalize_features(protein_feature)
    datasetF['pp'] = {'matrix': protein_sim_matrix1, 'edges': protein_sim_edge_index, 'feature': protein_feature}

    dp_matrix = np.vstack((
        np.hstack((np.zeros((drug_num, drug_num)), drug_protein_matrix)),
        np.hstack((drug_protein_matrix.T, np.zeros((protein_num, protein_num))))))
    dp_matrix = preprocess_adj(dp_matrix)
    dp_edge_index = get_edge_index(dp_matrix)
    dp_feature = np.vstack((np.hstack((drug_feature, drug_protein_matrix)),
                            np.hstack((drug_protein_matrix.T, protein_feature))))
    dp_feature = normalize_features(dp_feature)
    datasetF['dp'] = {'matrix': dp_matrix, 'edges': dp_edge_index,
                      'feature': dp_feature}

    ddpp_matrix = np.vstack((np.hstack((drug_sim_matrix, drug_protein_matrix)),
                             np.hstack((drug_protein_matrix.T, protein_sim_matrix))))
    ddpp_matrix = preprocess_adj(ddpp_matrix)
    ddpp_edge_index = get_edge_index(ddpp_matrix)

    datasetF['ddpp'] = {'matrix': ddpp_matrix, 'edges': ddpp_edge_index,
                        'feature': dp_feature}

    traindata_length = len(train_dataset)
    devdata_length = len(dev_dataset)

    train_dataset = torch.from_numpy(train_dataset)

    dev_dataset = torch.from_numpy(dev_dataset)
    test_dataset = torch.from_numpy(test_dataset)

    """Set a model."""
    ATTENTION = attention  # 'IA'

    # num_training_steps = len(train_loader) * iteration
    print('num_training_steps:', iteration)
    print('----------')

    AUCs_title = ('Epoch\t(es)\tTime\tLoss_train\tTrain_acc\t'
                  'Dev_result(dev_acc, dev_auc, dev_aupr, dev_precision, dev_recall, dev_f1, dev_mcc)\t')

    print(AUCs_title)

    model = DTI_pre(drug_num, protein_num, dim, layer_output, layer_IA, nhead, dropout, attention=ATTENTION).to(device)

    trainer = Trainer(model)
    tester = Tester(model)

    start = timeit.default_timer()

    # logging info
    MODEL_NAME = 'MIDTI'
    logging.info('MODEL: {}'.format(MODEL_NAME))
    logging.info('DATASET: {}'.format(DATASET))
    logging.info('TRAIN_DATASET_LENGTH: {}'.format(traindata_length))
    logging.info('DEV_DATASET_LENGTH: {}'.format(devdata_length))
    logging.info('DRUG_NUM: {}'.format(drug_num))
    logging.info('PROTEIN_NUM: {}'.format(protein_num))

    logging.info('ATTENTIOM: {}'.format(ATTENTION))
    logging.info('OPTIMIZER: {}'.format(optimizer))
    logging.info('DIM: {}'.format(dim))
    logging.info('MAX_EPOCHS: {}'.format(iteration))
    logging.info('attention_LAYERS: {}'.format(layer_IA))
    logging.info('learning rate: {}'.format(lr))
    logging.info('fold: {}'.format(fold))
    logging.info('random_seed: {}'.format(random_seed))

    best_auc = 0

    es = 0  # early stopping counter

    log_header = 'MIDTI Version:\n1.DATASET={}\n' \
                 '2. MODEL_NAME={}\n' \
                 '3. attention={} '\
                 '4. optimizer={}\n' \
                 '5. iteration={}\n' \
                 '6. atttention_layers={}\n' \
                 '7. learning rate={}\n' \
                 '8. dim = {}\n' \
                 '9. random_seed={}\n'.format(DATASET, MODEL_NAME, ATTENTION, optimizer, iteration, layer_IA, lr, dim, random_seed)

    if log_write:
        log_dir = '../output/log/'
        file_name = 'MIDTI_IA{}_fold{}'.format(layer_IA, fold) + '.log'

        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        f_path = os.path.join(log_dir, file_name)
        with open(f_path, 'a+') as f:
            f.write(log_header)


    for epoch in range(0, iteration):

        print('---------epoch{}---------'.format(epoch))
        loss_train, _ = trainer.train(train_dataset, datasetF)
        _, _, train_acc, _, _, _ , _, _, _= tester.dev(epoch, train_dataset, datasetF)  
        test_labels, test_scores, dev_acc, dev_auc, dev_aupr, dev_precision, dev_recall, dev_f1, dev_mcc = tester.dev(epoch, dev_dataset, datasetF) 

        end = timeit.default_timer()
        time = end - start
        AUCs = [epoch, (es), time, loss_train, train_acc,
                (dev_acc, dev_auc, dev_aupr, dev_precision, dev_recall, dev_f1, dev_mcc)]

        if log_write:
            tester.save_AUCs(AUCs, f_path)
        print(AUCs_title)
        print('\t'.join(map(str, AUCs)))

        if dev_auc > best_auc:
            best_epoch, best_loss_train, best_train_acc, best_acc, best_auc, best_aupr, best_precision, best_recall, best_f1, best_mcc = epoch, loss_train, train_acc, dev_acc, dev_auc, dev_aupr, dev_precision, dev_recall, dev_f1, dev_mcc
            es = 0  # early stop mechanism
            print('Get higher performance')

        elif dev_auc <= best_auc:
            es += 1
            if best_train_acc > 0.93 and es > 50:
                print('Early stopping counter reaches to 50, the training will stop')
                best_AUC = best_epoch, best_loss_train, best_train_acc, (
                best_acc, best_auc, best_aupr, best_precision, best_recall, best_f1, best_mcc)

                # np.savetxt('test_scores_fold{}.txt'.format(fold), test_scores)  
                # np.savetxt('test_labels_fold{}.txt'.format(fold), test_labels)
                print('the new best model', best_AUC)


                if log_write:
                    tester.save_AUCs(best_AUC, f_path)

                if best_auc > save_auc:
                    save_dir = '../output/model/MIDTI_{}_{}'.format(ATTENTION, dim)
                    if not os.path.isdir(save_dir):
                        os.makedirs(save_dir)

                    model_filename = ('{}-fold{}--{:.4f}.pkl'.format(DATASET, fold, best_auc))
                    model_path = os.path.join(save_dir, model_filename)
                    torch.save(model.state_dict(), model_path)

                    print(r'Saved the new best model (dev auc: {}to {}'.format(dev_auc, model_path))

                break
            elif es > 199:
                best_epoch, best_loss_train, best_train_acc, best_acc, best_auc, best_aupr, best_precision, best_recall, best_f1, best_mcc = epoch, loss_train, train_acc, dev_acc, dev_auc, dev_aupr, dev_precision, dev_recall, dev_f1, dev_mcc
                es = 0
            elif epoch == iteration - 1:
                best_AUC = best_epoch, best_loss_train, best_train_acc, (
                    best_acc, best_auc, best_aupr, best_precision, best_recall, best_f1, best_mcc)
                print('the new best model', best_AUC)
    return best_acc, best_auc, best_aupr, best_precision, best_recall, best_f1, best_mcc




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(message)s')

    """Hyperparameters."""
    nfold = 5
    DATASET = 'LuoDTI'
    dim = 512


    layer_IA = 3 #4#2 #1
    lr = 0.1 #0.05  #0.01  # 0.15 #0.20
    weight_decay = 5e-4 #5e-4
    iteration = 2000
    random_seed = 2021
    optimizer = 'lookahead-SGD'
    layer_output = 3  # mlp layer
    nhead = 8 #1 #2 #4 #8#16
    dropout = 0.1
    attention = 'IA'

    (dim, layer_output, layer_IA,
     iteration, random_seed, nhead) = map(int, [dim,layer_output, layer_IA,
     iteration, random_seed, nhead])
    lr, weight_decay, dropout = map(float, [lr, weight_decay, dropout])

    config = {'dim':dim, 'layer_output':layer_output, 'layer_IA':layer_IA,
     'iteration':iteration, 'nhead': nhead, 'lr':lr, 'weight_decay':weight_decay, 'dropout':dropout,'optimizer':optimizer}

    """CPU or GPU."""
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = torch.device('cuda:0')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')
    # device = torch.device('cpu')
    # print('The code uses CPU!!!')


    result_acc = np.zeros((nfold))
    result_auc = np.zeros((nfold))
    result_aupr = np.zeros((nfold))
    result_pre = np.zeros((nfold))
    result_recall = np.zeros((nfold))
    result_f1 = np.zeros((nfold))
    result_mcc = np.zeros((nfold))
    for fold in range(0, nfold):  
        # DATASET = 'LuoDTI'
        # print(device)
        best_acc, best_auc, best_aupr, best_precision, best_recall, best_f1, best_mcc = train(DATASET, fold, 0.9, 'IA', random_seed, True)
        result_acc[fold] = best_acc
        result_auc[fold] = best_auc
        result_aupr[fold] = best_aupr
        result_pre[fold] = best_precision
        result_recall[fold] = best_recall
        result_f1[fold] =best_f1
        result_mcc[fold] =best_mcc
    print('result_acc',result_acc)
    print('result_auc',result_auc)
    print('result_aupr',result_aupr)
    print('result_pre',result_pre)
    print('result_recall',result_recall)
    print('result_f1',result_f1)
    print('result_mcc',result_mcc)
    print('avg result_acc:', result_acc.mean(), result_acc.std())
    print('avg result_auc:', result_auc.mean(), result_auc.std())
    print('avg result_aupr:', result_aupr.mean(), result_aupr.std())
    print('avg result_pre:', result_pre.mean(), result_pre.std())
    print('avg result_recall:', result_recall.mean(), result_recall.std())
    print('avg result_f1:', result_f1.mean(), result_f1.std())
    print('avg result_mcc:', result_mcc.mean(), result_mcc.std())

