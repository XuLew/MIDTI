import csv

import numpy as np
import torch
from torch.nn.functional import normalize
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader


class dataClass(Dataset):
    def __init__(self, data_list_k):        
        self.idx0 = torch.LongTensor(data_list_k[:,0])
        self.idx1 = torch.LongTensor(data_list_k[:,1])
        self.labels = torch.LongTensor(data_list_k[:,2])
    def __len__(self):
        return len(self.idx0)
    def __getitem__(self, index):
        index0 = self.idx0[index]
        index1 = self.idx1[index]
        y = self.labels[index]
        return index0, index1, y

def getLoader(batch_size, data_list_k):
    params = {'batch_size': batch_size,
              'shuffle': True,
              'drop_last' :False}
    data_set = dataClass(data_list_k)
    loader = DataLoader(data_set, **params)
    return loader

def read_txt(path, delim):
    reader = np.loadtxt(path, dtype=int, delimiter=delim)
    # print(reader)
    md_data = []
    md_data += [[float(i) for i in row] for row in reader]
    return np.array(md_data)
    # return torch.Tensor(md_data)

def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        return np.array(md_data)
        # return torch.Tensor(md_data)

def outputCSVfile(filename, data):
    csvfile = open(filename, 'w', newline="")
    writer = csv.writer(csvfile)
    writer.writerows(data)  
    csvfile.close()

