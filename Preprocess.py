import math
import numpy as np
import scipy

import os
import scipy.io as sciio
import sys
import torch
from opts import get_opts
from scipy.signal import butter, filtfilt
from sklearn.decomposition import FastICA
from sklearn.model_selection import KFold

from tools import *

def DivideDataset(person_num=54):
    x = np.arange(person_num)       # 54表示人数
    kf = KFold(n_splits=10, shuffle=False)
    train_list, val_list, test_list = [], [], []
    for train_index, test_index in kf.split(x):
        train_list.append(train_index[int(person_num*0.1)::])
        val_list.append(train_index[:int(person_num*0.1)])
        test_list.append(test_index)
    return train_list, val_list, test_list

def Filter_EEG(eeg, f1=0.5, f2=40, fs = 200):
    [b1, a1] = butter(5, [f1/fs * 2, f2/fs * 2], 'bandpass')  # 截止频率*2/采样频率, 带通0.5-40
    eeg = filtfilt(b1, a1, eeg) # 数字滤波器
    
    return eeg

def Standardization(EEG_data, EEG_NOS_data):
    if torch.is_tensor(EEG_data):
        for i in range(EEG_data.shape[0]):
            EEG_data[i] = EEG_data[i] - torch.mean(EEG_data[i])
            EEG_NOS_data[i] = EEG_NOS_data[i] - torch.mean(EEG_NOS_data[i])
            
            EEG_data[i] = EEG_data[i] / torch.std(EEG_NOS_data[i], unbiased=False)
            EEG_NOS_data[i] = EEG_NOS_data[i] / torch.std(EEG_NOS_data[i], unbiased=False)
    else:
        for i in range(EEG_data.shape[0]):
            EEG_data[i] = EEG_data[i] - np.mean(EEG_data[i])
            EEG_NOS_data[i] = EEG_NOS_data[i] - np.mean(EEG_NOS_data[i])
            
            EEG_data[i] = EEG_data[i] / np.std(EEG_NOS_data[i])
            EEG_NOS_data[i] = EEG_NOS_data[i] / np.std(EEG_NOS_data[i])
    return EEG_data, EEG_NOS_data


def LoadEEGData(EEG_path, NOS_path, fold):
    data_signals = scipy.io.loadmat(EEG_path)
    noisy_artifact = scipy.io.loadmat(NOS_path)

    del data_signals['__header__'], data_signals['__version__'], data_signals['__globals__']
    del noisy_artifact['__header__'], noisy_artifact['__version__'], noisy_artifact['__globals__']

    train_list, val_list, test_list = DivideDataset(person_num=54)
    train_id, val_id, test_id = train_list[fold], val_list[fold], test_list[fold]

    fix_length = 5500
    train_data, train_noisy = [], []
    for n in train_id:
        n = n + 1
        reference = data_signals[f'sim{n}_resampled']
        data_artifact = noisy_artifact[f'sim{n}_con']
        shapes = data_artifact.shape
        if shapes[1] < fix_length:
            data_artifact = np.concatenate([data_artifact, torch.zeros(size=(shapes[0], fix_length - shapes[1]))], axis=1)
            reference = np.concatenate([reference, torch.zeros(size=(shapes[0], fix_length - shapes[1]))], axis=1)
        else:
            data_artifact = data_artifact[:, :fix_length]
            reference = reference[:, :fix_length]
        print(reference.shape)
  
        for c in range(reference.shape[0]):
            reference[c], data_artifact[c] = Filter_EEG(reference[c], fs = 200), Filter_EEG(data_artifact[c], fs = 200)     
        reference, data_artifact = Standardization(reference, data_artifact) 
            
        train_data.append(reference)
        train_noisy.append(data_artifact)

    EEG_train_data, NOS_train_data = train_data, train_noisy

    val_data, val_noisy = np.zeros((19, 10 * len(val_id), 500)), np.zeros((19, 10 * len(val_id), 500))
    for i in range(len(val_id)):
        n = val_id[i] + 1
        reference = data_signals[f'sim{n}_resampled']
        data_artifact = noisy_artifact[f'sim{n}_con']

        reference, data_artifact = reference[:, 0:5000], data_artifact[:, 0:5000]
        
        for c in range(reference.shape[0]):
            reference[c], data_artifact[c] = Filter_EEG(reference[c], fs = 200), Filter_EEG(data_artifact[c], fs = 200)
        reference, data_artifact = Standardization(reference, data_artifact) 
            
        reference, data_artifact = reference.reshape(19, -1, 500), data_artifact.reshape(19, -1, 500)
        val_data[:, i * 10:(i + 1) * 10, :] = reference
        val_noisy[:, i * 10:(i + 1) * 10, :] = data_artifact

    EEG_val_data, NOS_val_data = val_data, val_noisy

    test_data, test_noisy = np.zeros((19, 10 * len(test_id), 500)), np.zeros((19, 10 * len(test_id), 500))
    for i in range(len(test_id)):
        n = test_id[i] + 1
        reference = data_signals[f'sim{n}_resampled']
        data_artifact = noisy_artifact[f'sim{n}_con']

        reference, data_artifact = reference[:, 0:5000], data_artifact[:, 0:5000]

        for c in range(reference.shape[0]):
            reference[c], data_artifact[c] = Filter_EEG(reference[c], fs = 200), Filter_EEG(data_artifact[c], fs = 200)
        reference, data_artifact = Standardization(reference, data_artifact) 
        
        reference, data_artifact = reference.reshape(19, -1, 500), data_artifact.reshape(19, -1, 500)
        test_data[:, i * 10:(i + 1) * 10, :] = reference
        test_noisy[:, i * 10:(i + 1) * 10, :] = data_artifact

    EEG_test_data, NOS_test_data = test_data, test_noisy
    return EEG_train_data, NOS_train_data, EEG_val_data, NOS_val_data, EEG_test_data, NOS_test_data

class GetEEGData(object):
    def __init__(self, EEG_data, NOS_data, batch_size=128):
        super(GetEEGData, self).__init__()
        self.EEG_data, self.NOS_data = EEG_data, NOS_data
        self.batch_size = batch_size

    def len(self):
        return math.ceil(self.EEG_data.shape[1] / self.batch_size)     # ceil

    def get_item(self, item):
        EEG_data = self.EEG_data[:, item, :]
        NOS_data = self.NOS_data[:, item, :]
        return NOS_data, EEG_data

    def get_batch(self, batch_id):
        start_id, end_id = batch_id * self.batch_size, min((batch_id + 1) * self.batch_size, self.EEG_data.shape[1])
        EEG_NOS_batch, EEG_batch = [], []
        for item in range(start_id, end_id):
            EEG_NOS_data, EEG_data = self.get_item(item)
            EEG_NOS_batch.append(EEG_NOS_data), EEG_batch.append(EEG_data)
        EEG_NOS_batch, EEG_batch = np.array(EEG_NOS_batch), np.array(EEG_batch)
        
        # EEG_batch, EEG_NOS_batch = Standardization(EEG_batch, EEG_NOS_batch) 
        return EEG_NOS_batch, EEG_batch

class GetEEGData_train(object):
    def __init__(self, EEG_data, NOS_data, batch_size=128, device='cuda:0'):
        super(GetEEGData_train, self).__init__()
        self.device = device
        self.EEG_list = torch.Tensor(np.concatenate((np.array(EEG_data), np.array(EEG_data)), axis=0)).to(self.device)
        self.NOS_list = torch.Tensor(np.concatenate((np.array(NOS_data), np.array(EEG_data)), axis=0)).to(self.device)
        # self.EEG_list = torch.Tensor(np.array(EEG_data)).to(self.device)
        # self.NOS_list = torch.Tensor(np.array(NOS_data)).to(self.device)
        self.batch_size = batch_size
        self.random_shuffle()

    def len(self):
        return math.floor(self.start_point_idxs.shape[0] / self.batch_size)  # ceil

    def get_batch(self, batch_id):
        start_id, end_id = batch_id * self.batch_size, min((batch_id + 1) * self.batch_size, self.start_point_idxs.shape[0])
        start_point_batch = self.start_point_idxs[start_id:end_id]
        sample_batch = self.sample_idxs[start_id:end_id]
        EEG_samples = self.EEG_list[sample_batch, ...]
        NOS_samples = self.NOS_list[sample_batch, ...]
        gather_idx = torch.Tensor(np.array(list(range(500)))).to(self.device).long().unsqueeze(0).unsqueeze(0)
        gather_idx = gather_idx.repeat((EEG_samples.shape[0], EEG_samples.shape[1], 1))
        gather_idx = gather_idx + start_point_batch.unsqueeze(-1).unsqueeze(-1)
        EEG_batch = EEG_samples.gather(index=gather_idx, dim=2)
        EEG_NOS_batch = NOS_samples.gather(index=gather_idx, dim=2)
        
        # EEG_batch, EEG_NOS_batch = Standardization(EEG_batch, EEG_NOS_batch) 
        return EEG_NOS_batch, EEG_batch

    def random_shuffle(self):
        num_per_epoch_sample = 100
        self.start_point_idxs, self.sample_idxs = [], []
        for i in range(self.EEG_list.shape[0]):
            idx = np.random.permutation(self.EEG_list.shape[2] - 500)[:num_per_epoch_sample]
            self.start_point_idxs.append(idx), self.sample_idxs.append(np.zeros(shape=(num_per_epoch_sample,)) + i)
        self.start_point_idxs = np.concatenate(self.start_point_idxs, axis=0)
        self.sample_idxs = np.concatenate(self.sample_idxs, axis=0)
        shuffle = np.random.permutation(self.start_point_idxs.shape[0])
        self.start_point_idxs = torch.Tensor(self.start_point_idxs[shuffle]).to(self.device).long()
        self.sample_idxs = torch.Tensor(self.sample_idxs[shuffle]).to(self.device).long()


if __name__ == '__main__':
    noise_path = 'C:/Users/Administrator/Desktop/01_SSA-CCA multichannel/data/Contaminated_Data.mat'
    eeg_path = 'C:/Users/Administrator/Desktop/01_SSA-CCA multichannel/data/Pure_Data.mat'
    
    EEG_train_data, NOS_train_data, EEG_val_data, NOS_val_data, EEG_test_data, NOS_test_data = LoadEEGData(eeg_path, noise_path, 1)
    
    EEG_train_data = np.array(EEG_train_data)
    print(EEG_train_data.shape) # (43, 19, 5500)
