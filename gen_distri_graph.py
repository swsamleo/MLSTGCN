# For relative import
import os
import sys
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJ_DIR)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets.parking import *
import scipy.stats
from sklearn.preprocessing import normalize

from scipy.stats import wasserstein_distance


def KL_divergence(p,q):
    return scipy.stats.entropy(p, q)


raw_data = os.path.join(PROJ_DIR, 'data/distribution_240min.csv')

node_num = 40
data = pd.read_csv(raw_data).values


A_distri = np.zeros((node_num, node_num))

A_distri_ws = np.zeros((node_num, node_num))


for i in range(node_num):
    for j in range(node_num):
        ii = data[i]
        jj = data[j]

        kl = KL_divergence(ii, jj)

        ws = wasserstein_distance(ii, jj)

        A_distri[i, j] = kl

        A_distri_ws[i, j] = ws

A_distri = normalize(A_distri, axis=0, norm='l1')
A_distri_ws = normalize(A_distri_ws, axis=0, norm='l1')


np.save('/Users/wangshuo/data/traffic/graph/graph/distri_kl.npy', A_distri)
np.save('/Users/wangshuo/data/traffic/graph/graph/distri_ws.npy', A_distri_ws)
