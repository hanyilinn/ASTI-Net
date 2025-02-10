import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

class my_dataset(Dataset):  # 继承自Dataset类，Dataset是PyTorch用于表示数据集的抽象类

    def __init__(self, input, output, label, transform=None):
        self.input = input # ndarray
        self.output = output
        self.label = label
        self.transform = transform

    # 定义了当使用索引访问类实例时的行为，例如dataset[i]。它接受一个索引index并返回对应的数据和标签
    def __getitem__(self, index):  
        # 从self.data中选取第index行的数据，将其转换为PyTorch张量
        input = torch.from_numpy(self.input[index, :, :]).reshape(1, self.input.shape[1], self.input.shape[2]).float()
        output = torch.from_numpy(self.output[index, :, :]).reshape(1, self.output.shape[1], self.output.shape[2]).float()
        label = torch.from_numpy(self.label[index, :])

        return input, output, label

    def __len__(self):
        return self.input.shape[0]
    
class my_dataset_nolabel(Dataset):  # 继承自Dataset类，Dataset是PyTorch用于表示数据集的抽象类

    def __init__(self, input, output, transform=None):
        self.input = input # ndarray
        self.output = output
        self.transform = transform

    # 定义了当使用索引访问类实例时的行为，例如dataset[i]。它接受一个索引index并返回对应的数据和标签
    def __getitem__(self, index):  
        # 从self.data中选取第index行的数据，将其转换为PyTorch张量
        input = torch.from_numpy(self.input[index, :, :]).reshape(1, self.input.shape[1], self.input.shape[2]).float()
        output = torch.from_numpy(self.output[index, :, :]).reshape(1, self.output.shape[1], self.output.shape[2]).float()

        return input, output

    def __len__(self):
        return self.input.shape[0]


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True

# 计算标签张量元素相似率
def Ratio_of_SameElem(label, pre_label):
    # 检查两个张量的形状是否一致
    if label.shape != pre_label.shape:
        raise ValueError("Both tensors must have the same shape!")

    same_elements = torch.eq(label, pre_label)
    count = torch.sum(same_elements)
    total_elements = label.numel()  # 等同于 label.num_elements()
    ratio = count.float() / total_elements
    
    return ratio.item()  # 返回一个Python float


def cal_ACC_npy(predict, truth):
    # 计算ACC，针对二维信号
    vy_ = predict - np.mean(predict, axis=(1, 2)).reshape((predict.shape[0], 1, 1))
    vy = truth - np.mean(truth, axis=(1, 2)).reshape((truth.shape[0], 1, 1))
    cc = np.sum(vy_ * vy, axis=(1, 2)) / (np.sqrt(np.sum(vy_ ** 2, axis=(1, 2))) * np.sqrt(np.sum(vy ** 2, axis=(1, 2))) + 1e-8)
    average_cc = np.mean(cc)
    return average_cc


def cal_ACC_tensor(predict, truth):
    # 计算ACC，针对二维信号
    vy_ = predict - torch.mean(predict, dim=(1, 2)).unsqueeze(1).unsqueeze(2)
    vy = truth - torch.mean(truth, dim=(1, 2)).unsqueeze(1).unsqueeze(2)
    cc = torch.sum(vy_ * vy, dim=(1, 2)) / (
            torch.sqrt(torch.sum(vy_ ** 2, dim=(1, 2))) * torch.sqrt(torch.sum(vy ** 2, dim=(1, 2))) + 1e-8)
    average_cc = torch.mean(cc)
    return average_cc

def ACCLoss(predict, truth):
    # 计算ACC损失，针对二维信号
    vy_ = predict - torch.mean(predict, dim=(1, 2)).unsqueeze(1).unsqueeze(2)
    vy = truth - torch.mean(truth, dim=(1, 2)).unsqueeze(1).unsqueeze(2)
    cc = torch.sum(vy_ * vy, dim=(1, 2)) / (
            torch.sqrt(torch.sum(vy_ ** 2, dim=(1, 2))) * torch.sqrt(torch.sum(vy ** 2, dim=(1, 2))) + 1e-8)
    average_cc = torch.mean(cc)
    return torch.ones_like(average_cc) - average_cc


def cal_RRMSE_tensor(predict, truth):
    # 计算RRMSE，针对二维信号
    l1 = (predict - truth) ** 2
    lo = torch.sum(l1, dim=(1, 2)) / (predict.shape[1] * predict.shape[2])
    l3 = truth ** 2
    l4 = torch.sum(l3, dim=(1, 2)) / (truth.shape[1] * truth.shape[2])
    rrmse = torch.sqrt(lo) / torch.sqrt(l4)
    rrmse = torch.mean(rrmse)
    return rrmse


def cal_SNR(predict, truth):
    if torch.is_tensor(predict):
        predict = predict.detach().cpu().numpy()
    if torch.is_tensor(truth):
        truth = truth.detach().cpu().numpy()

    PS = np.sum(np.square(truth), axis=(1, 2))  # 信号功率
    PN = np.sum(np.square(predict - truth), axis=(1, 2))  # 噪声功率
    ratio = PS / PN
    snr = np.mean(10 * np.log10(ratio))
    return snr


def calculate_accuracy(labels, predictions):
    correct = torch.eq(labels, predictions).float()
    return torch.mean(correct).item()

def calculate_precision_recall_f1(labels, predictions):
    TP = torch.sum((predictions == 1) & (labels == 1)).float()
    FP = torch.sum((predictions == 1) & (labels == 0)).float()
    FN = torch.sum((predictions == 0) & (labels == 1)).float()
    TN = torch.sum((predictions == 0) & (labels == 0)).float()

    precision = TP / (TP + FP) if (TP + FP) != 0 else torch.tensor(0.0)
    recall = TP / (TP + FN) if (TP + FN) != 0 else torch.tensor(0.0)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else torch.tensor(0.0)

    return precision.item(), recall.item(), f1.item()


def calculate_auc(labels, predictions):
    # AUC 要求预测值是概率，如果不是概率而是0或1，需要使用不同的方法
    # 此处假设 predictions 已经是概率值，如果不是请将其转换为概率或者使用其他适合的评估方式
    labels_np = labels.numpy()
    predictions_np = predictions.numpy()
    auc = roc_auc_score(labels_np.flatten(), predictions_np.flatten())
    return auc

def evaluate_model(labels, pre_labels):
    # 假设 labels 和 pre_labels 都是形状为 (batch_size, 22) 的张量
    # 将二维张量转换为一维，以进行整体评估
    labels = labels.view(-1)
    pre_labels = pre_labels.view(-1)

    # 计算准确率
    accuracy = calculate_accuracy(labels, pre_labels)
    
    # 计算精确率、召回率、F1分数
    precision, recall, f1 = calculate_precision_recall_f1(labels, pre_labels)
    
    # 计算AUC值
    auc = calculate_auc(labels, pre_labels)
    
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "AUC": auc
    }
