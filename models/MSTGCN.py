# -*- coding:utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import torch
import torch.utils.data
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from scipy.sparse.linalg import eigs

from torch_geometric.utils import dense_to_sparse, get_laplacian, to_dense_adj


def cheb_polynomial_torch(L_tilde, K):
    N = L_tilde.shape[0]

    cheb_polynomials = [torch.eye(N).to(L_tilde.device), L_tilde.clone()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials


def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])


class cheb_conv(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, K, fusiongraph, in_channels, out_channels, device):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv, self).__init__()
        self.K = K
        self.fusiongraph = fusiongraph
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = device
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])

    def forward(self, x):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        adj_for_run = self.fusiongraph()

        edge_idx, edge_attr = dense_to_sparse(adj_for_run)
        # edge_idx_l, edge_attr_l = get_laplacian(edge_idx, edge_attr, 'sym')
        edge_idx_l, edge_attr_l = get_laplacian(edge_idx, edge_attr)

        L_tilde = to_dense_adj(edge_idx_l, edge_attr=edge_attr_l)[0]
        cheb_polynomials = cheb_polynomial_torch(L_tilde, self.K)

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):

                T_k = cheb_polynomials[k]  # (N,N)

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = graph_signal.permute(0, 2, 1).matmul(T_k).permute(0, 2, 1)

                output = output + rhs.matmul(theta_k)

            outputs.append(output.unsqueeze(-1))

        return F.relu(torch.cat(outputs, dim=-1))


class MSTGCN_block(nn.Module):

    def __init__(self, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, fusiongraph, device):
        super(MSTGCN_block, self).__init__()
        self.cheb_conv = cheb_conv(K, fusiongraph, in_channels, nb_chev_filter, device)
        self.time_conv = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3), stride=(1, time_strides), padding=(0, 1))
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.ln = nn.LayerNorm(nb_time_filter)

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, nb_time_filter, T)
        '''
        # cheb gcn
        spatial_gcn = self.cheb_conv(x)  # (b,N,F,T)

        # convolution along the time axis
        time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3))  # (b,F,N,T)

        # residual shortcut
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))  # (b,F,N,T)

        x_residual = self.ln(F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)  # (b,N,F,T)

        return x_residual


class MSTGCN_submodule(nn.Module):

    def __init__(self, gpu_id, fusiongraph, in_channels, len_input, num_for_predict):


    # def __init__(self, DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials, num_for_predict, len_input):
        '''
        :param nb_block:
        :param in_channels:
        :param K:
        :param nb_chev_filter:
        :param nb_time_filter:
        :param time_strides:
        :param cheb_polynomials:
        :param nb_predict_step:
        '''

        # Fusion Graph
        device = 'cuda:%d' % gpu_id
        DEVICE = device

        # -----------------

        # Parameters
        K = 3
        nb_block = 2
        nb_chev_filter = 64
        nb_time_filter = 64
        time_strides = 1
        num_of_vertices = fusiongraph.graph.node_num


        super(MSTGCN_submodule, self).__init__()

        self.BlockList = nn.ModuleList([MSTGCN_block(in_channels, K, nb_chev_filter, nb_time_filter, time_strides, fusiongraph, device)])

        self.BlockList.extend([MSTGCN_block(nb_time_filter, K, nb_chev_filter, nb_time_filter, 1, fusiongraph, device) for _ in range(nb_block-1)])

        self.final_conv = nn.Conv2d(int(len_input/time_strides), num_for_predict, kernel_size=(1, nb_time_filter))

        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''

        x = x.permute((0, 2, 3, 1))

        for block in self.BlockList:
            x = block(x)

        output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)

        output = output.permute((0, 2, 1))[..., None]

        return output
