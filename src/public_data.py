import pandas as pd
import numpy as np
import os
import torch
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler


def load_CICIDS(num_benign, num_malic, action_node_idx):
    csv_file = '../datasets/public/CICIDS-2017.csv'
    if not os.path.exists(csv_file):
        print('The dataset file does not exist.')

    df = pd.read_csv(csv_file, low_memory=False)
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    x = scaler.fit_transform(x)
    x = torch.from_numpy(x).float()

    attack_types = [ 'Web Attack - Brute Force','DoS slowloris','FTP-Patator', 'SSH-Patator','DDoS', 'Bot', 'PortScan']
    num_action_nodes = len(action_node_idx)
    total_benign = num_benign * num_action_nodes + num_malic * num_action_nodes * num_action_nodes
    benign = x[y == 'BENIGN'][: total_benign].reshape(-1, num_action_nodes, x.shape[1])
    x_benign = benign[:num_benign].clone()
    y_benign = torch.zeros(x_benign.shape[0], x_benign.shape[1])

    x_malic = benign[num_benign:num_benign + num_malic * num_action_nodes].clone()
    y_malic = torch.zeros(x_malic.shape[0], x_malic.shape[1])

    for idx, val in enumerate(action_node_idx):
        attack = attack_types[idx]
        x_malic[idx * num_malic:(idx+1) * num_malic, idx, :] = x[y == attack][:num_malic]
        y_malic[idx * num_malic:(idx+1) * num_malic, idx] = 1

    return x_benign, y_benign, x_malic, y_malic

def gene_dataset(action_node_idx, num_nodes, num_benign, num_malic):
    """Generate Dataset 2
    """
    num_action_nodes = len(action_node_idx)
    x_benign, y_benign, x_malic, y_malic = load_CICIDS(num_benign, num_malic, action_node_idx)
    rt_meas_dim = x_benign.shape[2]
    X_benign = torch.zeros(num_benign, num_nodes, rt_meas_dim)
    X_benign[:, action_node_idx, :] = x_benign
    Y_benign = y_benign

    X_malic = torch.zeros(num_malic * num_action_nodes, num_nodes, rt_meas_dim)
    X_malic[:, action_node_idx, :] = x_malic
    Y_malic = y_malic

    X = torch.cat((X_benign, X_malic), dim=0)
    Y = torch.cat((Y_benign, Y_malic), dim=0)

    return X, Y
