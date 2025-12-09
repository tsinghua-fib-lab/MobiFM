import numpy as np
import torch as th
import json
import torch
import datetime
import copy
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer
import pickle
import torch.nn.functional as F
from matplotlib import pyplot as plt
import torch
import torch.nn as nn


class BoundedLogitScaler:
    def __init__(self, ranges_dict, cont_names, eps=1e-6):
        self.ranges = ranges_dict
        self.names = cont_names
        self.eps = eps
    def to_unbounded(self, x_cont):  # [B,8]
        outs = []
        for i, name in enumerate(self.names):
            lo, hi = self.ranges[name]
            u01 = (x_cont[:, i] - lo) / (hi - lo)
            u01 = torch.clamp(u01, self.eps, 1 - self.eps)
            outs.append(torch.log(u01) - torch.log(1 - u01))
        return torch.stack(outs, dim=-1)
    def to_bounded(self, x_star):    # [B,8]
        outs = []
        for i, name in enumerate(self.names):
            lo, hi = self.ranges[name]
            u01 = torch.sigmoid(x_star[:, i])
            outs.append(u01 * (hi - lo) + lo)
        return torch.stack(outs, dim=-1)

@torch.no_grad()
def encode_to_zC(flow, scaler, x_cont, d_oh, K_AUG):

    x_star = scaler.to_unbounded(x_cont)              # [B,8]
    a0 = torch.zeros(x_star.size(0), K_AUG)
    x_aug = torch.cat([x_star, a0], dim=-1)                      # [B,8+k]
    z_flow, _ = flow(x_aug, d_oh)                     # [B,8+k]
    zC = torch.cat([z_flow, d_oh], dim=-1)            # [B,C]
    return zC

@torch.no_grad()
def encode_to_zC_time(flow, scaler, x_cont, d_oh, K_AUG, NUM_CONT, C_FLOW):

    B, Cc, T = x_cont.shape
    _, K, T2 = d_oh.shape
    assert Cc == NUM_CONT and T2 == T, "temporal or channel mismatch"

    dev = x_cont.device

    x_flat = x_cont.permute(0, 2, 1).reshape(B*T, NUM_CONT)    # [B*T, NUM_CONT]
    d_flat = d_oh.permute(0, 2, 1).reshape(B*T, K)             # [B*T, K]


    x_star = scaler.to_unbounded(x_flat)                        # [B*T, NUM_CONT]
    a0 = torch.zeros(x_star.size(0), K_AUG, device=dev)         # [B*T, K_AUG]
    x_aug = torch.cat([x_star, a0], dim=-1)                     # [B*T, C_FLOW]


    z_flow, _ = flow(x_aug, d_flat)                             # [B*T, C_FLOW]


    z_flow_bt = z_flow.view(B, T, C_FLOW).permute(0, 2, 1)      # [B, C_FLOW, T]
    d_oh_bt   = d_flat.view(B, T, K).permute(0, 2, 1)           # [B, K, T]
    zC = torch.cat([z_flow_bt, d_oh_bt], dim=1)                 # [B, C_FLOW+K, T]
    return zC


def one_hot(values, choices):
    idx_map = {v: i for i, v in enumerate(choices)}
    K = len(choices)
    oh = np.zeros((len(values), K), dtype=float)
    for n, v in enumerate(values):
        i = idx_map.get(v, None)
        if i is not None:
            oh[n, i] = 1.0
    return oh

class STNet(nn.Module):
    def __init__(self, in_dim, cond_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + cond_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, in_dim * 2)
        )
    def forward(self, x_in, cond):
        h = torch.cat([x_in, cond], dim=-1)
        st = self.net(h); s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s) * 2.0
        return s, t

class RealNVPCoupling(nn.Module):
    def __init__(self, dim, cond_dim, mask):
        super().__init__()
        self.register_buffer("mask", mask)
        self.st = STNet(in_dim=dim, cond_dim=cond_dim)
    def forward(self, x, cond):
        x_a = x * self.mask
        s, t = self.st(x_a, cond)
        y = x_a + (1 - self.mask) * (x * torch.exp(s) + t)
        logdet = ((1 - self.mask) * s).sum(dim=-1)
        return y, logdet
    def inverse(self, y, cond):
        y_a = y * self.mask
        s, t = self.st(y_a, cond)
        x = y_a + (1 - self.mask) * ((y - t) * torch.exp(-s))
        logdet = -((1 - self.mask) * s).sum(dim=-1)
        return x, logdet


class ConditionalRealNVP(nn.Module):
    def __init__(self, dim, cond_dim, num_coupling=8):
        super().__init__()

        m0 = torch.tensor([1 if i % 2 == 0 else 0 for i in range(dim)], dtype=torch.float32)
        m1 = 1. - m0
        masks = [m0 if i % 2 == 0 else m1 for i in range(num_coupling)]
        self.layers = nn.ModuleList([RealNVPCoupling(dim, cond_dim, m) for m in masks])
    def forward(self, x, cond):
        z = x; sld = torch.zeros(x.size(0), device=x.device)
        for layer in self.layers:
            z, ld = layer(z, cond); sld = sld + ld
        return z, sld
    def inverse(self, z, cond):
        x = z; sld = torch.zeros(z.size(0), device=z.device)
        for layer in reversed(self.layers):
            x, ld = layer.inverse(x, cond); sld = sld + ld
        return x, sld

def sample_data_with_timestamp_and_features(data, timestamps, features, origin, ref, N, datatype, sample_length=96, total_len = 168):

    B, T = data.shape
    sampled_data = np.zeros((data.shape[0], N, sample_length))
    sampled_timestamps = np.zeros((timestamps.shape[0], N, sample_length, 2))
    sampled_features = np.zeros((features.shape[0], min(features.shape[1],sample_length), N, features.shape[-1]))
    sampled_origins = np.zeros((origin.shape[0], N, sample_length))
    sampled_ref  = np.zeros((ref.shape[0], sample_length, N, ref.shape[-1]))

    for i in range(data.shape[0]):  # 遍历所有 4000 个样本

        if 'RSRP' in datatype:
            random_indices = np.random.randint(0, 1, size=N)
        else:
            # 随机选择 N 个点
            random_indices = np.random.randint(0, total_len- sample_length, size=N)

        for j, idx in enumerate(random_indices):
            # 向后选择 sample_length 个点，并进行循环补充
            end_idx = idx + sample_length
            sampled_data[i, j, :] = data[i, idx:end_idx]
            sampled_origins[i, j, :] = origin[i, idx:end_idx]
            sampled_timestamps[i, j, :, :] = timestamps[i, idx:end_idx, :]

            # 将特征序列根据随机点扩展
            sampled_features[i, :, j, :] = features[i, :, :]
            sampled_ref[i, :, j, :] = ref[i, idx:end_idx, :]

    return sampled_data.reshape(B*N, sample_length), sampled_timestamps.reshape(B*N, sample_length, 2), sampled_features.reshape(B*N, features.shape[1], features.shape[-1]), sampled_origins.reshape(B*N, sample_length), sampled_ref.reshape(B*N, sampled_ref.shape[1], sampled_ref.shape[-1])




class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def cal_percentage(data, percentage):
    return torch.quantile(data, percentage)
    
def clip_data(data, threshold):
    return torch.where(data > threshold, threshold, data)

def normalize_cond(cond):
    # 计算 K 维度上的最小值和最大值，保持 N, L 维度
    min_vals = cond.min(axis=0, keepdims=True).min(axis=-1, keepdims=True)  # shape (1, K, 1)
    max_vals = cond.max(axis=0, keepdims=True).max(axis=-1, keepdims=True)  # shape (1, K, 1)

    # 避免除零错误（如果 min == max，则保持原值）
    normed = (cond - min_vals) / (max_vals - min_vals + 1e-8) * 2 - 1

    return normed

def data_load_single(args, datatype):
    

    X, T, C, ref = raw_load(args, datatype) # N, L
    flattened_data = X.flatten().numpy()
    Q1 = np.percentile(flattened_data, 25)
    Q3 = np.percentile(flattened_data, 95)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR


    outlier_indices = []
    for i in range(X.shape[0]):
        if torch.any((X[i] < lower_bound) | (X[i] > upper_bound)):
            outlier_indices.append(i)

    mask = torch.ones(X.shape[0], dtype=torch.bool)
    mask[outlier_indices] = False
    X = X[mask]
    T = T[mask]
    C = C[mask]
    ref = ref[mask]

    # print("数据集规模：", X.shape[0])
    # print("数据最大值：", X.max())
    # print("Clip后数据最大值：", X.max())



    train_idx, test_idx = train_test_split(np.arange(len(X)), test_size=0.3, random_state=42)
    val_idx = test_idx
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(X[train_idx].reshape(-1,1))



    data_scaled = scaler.transform(X.reshape(-1,1)).reshape(X.shape)
    ref = scaler.transform(ref.reshape(-1,1)).reshape(ref.shape)

    cond = C
    args.feature_size = cond.shape[-1]
    timestamp = T

    sample_data_train, sample_time_train, sample_cond_train, sample_X_train, sample_ref_train = sample_data_with_timestamp_and_features(data_scaled[train_idx], timestamp[train_idx], cond[train_idx], X[train_idx],ref[train_idx], args.sample, datatype)
    sample_data_test, sample_time_test, sample_cond_test, sample_X_test, sample_ref_test = sample_data_with_timestamp_and_features(data_scaled[test_idx], timestamp[test_idx], cond[test_idx], X[test_idx],ref[test_idx], 1, datatype)
    sample_data_valid, sample_time_valid, sample_cond_valid, sample_X_valid, sample_ref_valid = sample_data_with_timestamp_and_features(data_scaled[val_idx], timestamp[val_idx], cond[val_idx],  X[val_idx],ref[val_idx], 1, datatype)


    args.seq_len = sample_data_train.shape[-1]

    data_train = [[sample_data_train[i], sample_cond_train[i], sample_X_train[i], sample_time_train[i], sample_ref_train[i]] for i in range(sample_X_train.shape[0])]

    data_test = [[sample_data_test[i], sample_cond_test[i], sample_X_test[i], sample_time_test[i], sample_ref_test[i]] for i in range(sample_X_test.shape[0])]
    data_valid = [[sample_data_valid[i], sample_cond_valid[i], sample_X_valid[i], sample_time_valid[i], sample_ref_valid[i]] for i in range(sample_X_valid.shape[0])]


    train_dataset = MyDataset(data_train)
    val_dataset = MyDataset(data_valid)
    test_dataset = MyDataset(data_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)



    return  train_loader, test_loader, val_loader, scaler


def data_load(args):

    data_all = []
    test_data_all = []
    val_data_all = []
    my_scaler_all = {}

    for dataset_name in args.dataset.split('_'):
        data, test_data, val_data, my_scaler = data_load_single(args,dataset_name)
        data_all.append([dataset_name, data])
        test_data_all.append([dataset_name, test_data])
        val_data_all.append([dataset_name, val_data])
        my_scaler_all[dataset_name] = my_scaler

    data_all = [(name, i) for name, data in data_all for i in data]
    test_data_all = [(name, i) for name, test_data in test_data_all for i in test_data]
    val_data_all = [(name, i) for name, val_data in val_data_all for i in val_data]
    
    return data_all, test_data_all, val_data_all, my_scaler_all


def data_load_main(args):

    data, val_data, test_data, scaler = data_load(args)

    return data, test_data, val_data, scaler

def raw_load(args, datatype):
    if 'RSRP1' in datatype:
        CONT_NAMES = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"]
        RANGES = {
            "c1": (5.0, 20.0),
            "c2": (0.0, 1000.0),
            "c3": (0.0, 1000.0),
            "c4": (0.0, 500.0),
            "c5": (0.0, 500.0),
            "c6": (0.0, 1.0),
            "c7": (0.6, 1.0),
            "c8": (0.0, 1.0),
        }
        D1_LIST = [2585.0, 2604.8, 2624.6]  # 例：频点 MHz
        D2_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]  # 例：开关
        TARGET_C = 256
        K1, K2 = len(D1_LIST), len(D2_LIST)
        K = K1 + K2
        NUM_CONT = len(CONT_NAMES)  # 8
        K_AUG = max(0, TARGET_C - (NUM_CONT + K))
        C_FLOW = NUM_CONT + K_AUG

        path = './datasets/' + datatype + '.npz'
        file = np.load(path, allow_pickle=True)
        data = file['data']
        cond = file['cond']
        cal_c = file['calc']
        N, T = data.shape
        time_idx = torch.arange(T)
        # day = (time_idx // 24) % 4
        # hour = time_idx % 24
        day = torch.zeros_like(time_idx)
        hour = time_idx
        time_stamp = torch.stack([day, hour], dim=-1)  # (T, 2)
        time_stamp = time_stamp.unsqueeze(0).expand(N, -1, -1)
        data = torch.tensor(data, dtype=torch.float32)  # (N, L)


        time = time_stamp


        flow = ConditionalRealNVP(dim=C_FLOW, cond_dim=K, num_coupling=8)
        flow.load_state_dict(torch.load("MapModel1.pth"), strict=True)
        flow.eval()
        scaler = BoundedLogitScaler(RANGES, CONT_NAMES)
        c1 = torch.from_numpy(cond[:, 0]).float()
        c2 = torch.from_numpy(cond[:, 2]).float()
        c3 = torch.from_numpy(cond[:, 3]).float()
        c4 = torch.from_numpy(cond[:, 4]).float()
        c5 = torch.from_numpy(cond[:, 5]).float()
        c6 = torch.from_numpy(cond[:, 6]).float()
        c7 = torch.from_numpy(cond[:, 7]).float()
        c8 = torch.from_numpy(cond[:, 8]).float()
        c_cont = torch.stack([c1,c2,c3,c4,c5,c6,c7,c8], dim=-1).permute(0,2,1)
        d_oh = torch.zeros(N, K, T)
        for i in range(T):
            oh1 = torch.from_numpy(one_hot(cond[:, 1, i], D1_LIST).astype(np.float32))
            oh2 = torch.from_numpy(one_hot(cond[:, 10, i], D2_LIST).astype(np.float32))
            d_oh_ini = torch.cat([oh1, oh2], dim=-1)  # [N, K]
            d_oh[:,:,i] = d_oh_ini

        zC = encode_to_zC_time(flow, scaler, c_cont, d_oh, K_AUG, NUM_CONT, C_FLOW)  # [8, C]
        cond =  np.array(zC, dtype=np.float32).transpose(0,2,1)

        # RSP = cond[:, 0]
        # FB = cond[:, 1]
        # Ht = cond[:, 2]
        # Hr = cond[:, 3]
        # L = cond[:, 4]
        # D = cond[:, 5]
        # cosA = cond[:, 6]
        # cos_betaV = cond[:, 7]
        # cosC = cond[:, 8]
        # UCI = cond[:, 9]
        # CI = cond[:, 10]
        theory_value = np.array(cal_c, dtype=np.float32).transpose(0,2,1)
    elif 'RSRP2' in datatype:
        CONT_NAMES = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"]
        RANGES = {
            "c1": (-60, 0),
            "c2": (0.0, 1000.0),
            "c3": (0.0, 1000.0),
            "c4": (0.0, 500.0),
            "c5": (0.0, 600.0),
            "c6": (0.0, 1.0),
            "c7": (0.0, 1.0),
            "c8": (-1.0, 1.0),
        }
        D1_LIST = [2585.0, 2604.8]  # 例：频点 MHz
        D2_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 例：开关
        TARGET_C = 256
        K1, K2 = len(D1_LIST), len(D2_LIST)
        K = K1 + K2
        NUM_CONT = len(CONT_NAMES)  # 8
        K_AUG = max(0, TARGET_C - (NUM_CONT + K))
        C_FLOW = NUM_CONT + K_AUG

        path = './datasets/' + datatype + '.npz'
        file = np.load(path, allow_pickle=True)
        data = file['data']
        cond = file['cond']
        cal_c = file['calc']
        mask = ~np.isinf(cal_c).any(axis=1)
        cal_c = cal_c[mask]
        cond  = cond[mask]
        data = data[mask]
        cal_c =  np.expand_dims(cal_c, axis=1)
        N, T = data.shape
        time_idx = torch.arange(T)
        # day = (time_idx // 24) % 4
        # hour = time_idx % 24
        day = torch.zeros_like(time_idx)
        hour = time_idx
        time_stamp = torch.stack([day, hour], dim=-1)  # (T, 2)
        time_stamp = time_stamp.unsqueeze(0).expand(N, -1, -1)
        data = torch.tensor(data, dtype=torch.float32)  # (N, L)


        time = time_stamp


        flow = ConditionalRealNVP(dim=C_FLOW, cond_dim=K, num_coupling=8)
        flow.load_state_dict(torch.load("MapModel2.pth"), strict=True)
        flow.eval()
        scaler = BoundedLogitScaler(RANGES, CONT_NAMES)
        c1 = torch.from_numpy(cond[:, 0]).float()
        c2 = torch.from_numpy(cond[:, 2]).float()
        c3 = torch.from_numpy(cond[:, 3]).float()
        c4 = torch.from_numpy(cond[:, 4]).float()
        c5 = torch.from_numpy(cond[:, 5]).float()
        c6 = torch.from_numpy(cond[:, 6]).float()
        c7 = torch.from_numpy(cond[:, 7]).float()
        c8 = torch.from_numpy(cond[:, 8]).float()
        c_cont = torch.stack([c1,c2,c3,c4,c5,c6,c7,c8], dim=-1).permute(0,2,1)
        d_oh = torch.zeros(N, K, T)
        for i in range(T):
            oh1 = torch.from_numpy(one_hot(cond[:, 1, i], D1_LIST).astype(np.float32))
            oh2 = torch.from_numpy(one_hot(cond[:, 10, i], D2_LIST).astype(np.float32))
            d_oh_ini = torch.cat([oh1, oh2], dim=-1)  # [N, K]
            d_oh[:,:,i] = d_oh_ini

        zC = encode_to_zC_time(flow, scaler, c_cont, d_oh, K_AUG, NUM_CONT, C_FLOW)  # [8, C]
        cond =  np.array(zC, dtype=np.float32).transpose(0,2,1)

        # RSP = cond[:, 0]
        # FB = cond[:, 1]
        # Ht = cond[:, 2]
        # Hr = cond[:, 3]
        # L = cond[:, 4]
        # D = cond[:, 5]
        # cosA = cond[:, 6]
        # cos_betaV = cond[:, 7]
        # cosC = cond[:, 8]
        # UCI = cond[:, 9]
        # CI = cond[:, 10]
        theory_value = np.array(cal_c, dtype=np.float32).transpose(0,2,1)
    else:
        path = './datasets/' + datatype + '.npz'
        file = np.load(path, allow_pickle=True)
        data = file['data']#[idd]
        cond = np.load(f'./datasets/kg_embeddings_{datatype}.npy').reshape(-1,1,256)
        data = torch.tensor(data, dtype=torch.float32) # (N, L)
        cond = np.array(cond, dtype=np.float32) # (N, K, L)
        time = file['time']
        theory_value = np.expand_dims(np.zeros_like(data), axis=-1)#FOR ALIGNING WITH RSRP VECTOR SHAPE, MEANINGLESS


    return data, time, cond, theory_value