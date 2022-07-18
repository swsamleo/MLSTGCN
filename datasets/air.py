# For relative import
import os
import sys
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJ_DIR)

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import torch
from torch.utils import data
from util import *


class AirGraph():

    def __init__(self, graph_dir, config_graph, gpu_id):

        device = 'cuda:%d' % gpu_id

        use_graph, fix_weight = config_graph['use'], config_graph['fix_weight']
        tempp_diag_zero = config_graph['tempp_diag_zero']
        distri_type = config_graph['distri_type']

        self.A_dist = torch.from_numpy(np.float32(np.load(os.path.join(graph_dir, 'dist.npy')))).to(device)
        self.A_neighb = torch.from_numpy(np.float32(np.load(os.path.join(graph_dir, 'neigh.npy')))).to(device)
        self.A_func = torch.from_numpy(np.float32(np.load(os.path.join(graph_dir, 'func.npy')))).to(device)
        if distri_type == 'kl':
            self.A_distri = torch.from_numpy(np.float32(np.load(os.path.join(graph_dir, 'distri_kl.npy')))).to(device)
        elif distri_type == 'ws':
            self.A_distri = torch.from_numpy(np.float32(np.load(os.path.join(graph_dir, 'distri_kl.npy')))).to(device)
        else:
            self.A_distri = torch.from_numpy(
                np.float32(pd.read_csv(os.path.join(graph_dir, 'areaparacorr_92_air.csv'), header=None).values)).to(device)
        self.A_tempp = torch.from_numpy(np.float32(np.load(os.path.join(graph_dir, 'tempp_pm25.npy')))).to(device)

        self.node_num = self.A_dist.shape[0]

        if tempp_diag_zero:
            self.A_tempp.fill_diagonal_(0)

        self.use_graph = use_graph
        self.fix_weight = fix_weight
        self.graph_num = len(use_graph)

    def get_used_graphs(self):
        graph_list = []
        for name in self.use_graph:
            graph_list.append(self.get_graph(name))
        return graph_list

    def get_fix_weight(self):
        return (self.A_dist * 0.0829 + \
               self.A_neighb * 0.2050 + \
               self.A_distri * 0.1004 + \
               self.A_tempp * 0.5276 + \
               self.A_func * 0.0841) / 5

    def get_graph(self, name):
        if name == 'dist':
            return self.A_dist
        elif name == 'neighb':
            return self.A_neighb
        elif name == 'distri':
            return self.A_distri
        elif name == 'tempp':
            return self.A_tempp
        elif name == 'func':
            return self.A_func
        else:
            raise NotImplementedError


class Air(data.Dataset):

    def __init__(self, data_dir, data_type):
        assert data_type in ['train', 'val', 'test']
        self.data_type = data_type
        self._load_data(data_dir)

    def _load_data(self, data_dir):
        self.data = {}
        for category in ['train', 'val', 'test']:
            cat_data = np.load(os.path.join(data_dir, category + '.npz'))
            self.data['x_' + category] = cat_data['x'].astype(np.float32)
            self.data['y_' + category] = cat_data['y'].astype(np.float32)
        self.scaler = StandardScaler(mean=self.data['x_train'][..., 0].mean(), std=self.data['x_train'][..., 0].std())
        for category in ['train', 'val', 'test']:
            self.data['x_' + category][..., 0] = self.scaler.transform(self.data['x_' + category][..., 0])
        self.x, self.y = self.data['x_%s' % self.data_type], self.data['y_%s' % self.data_type]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


if __name__ == '__main__':
    graph_dir = 'data/graph'
    gpu_id = 0
    use_graph = ['dist']
    graph = AirGraph(graph_dir, use_graph, gpu_id)
    data_dir = 'data/temporal_data'
    parking = AirGraph(data_dir=data_dir, data_type='train')
    print(len(parking))
    parking = AirGraph(data_dir=data_dir, data_type='val')
    print(len(parking))
    parking = AirGraph(data_dir=data_dir, data_type='test')
    print(len(parking))
