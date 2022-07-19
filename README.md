# Long-term Spatio-Temporal Forecasting via Dynamic Multiple-Graph Attention

## Introduction

We propose a new dynamic multi-graph fusion module to characterize the correlations of nodes within a graph and the nodes across graphs via the spatial attention and graph attention mechanisms. Furthermore, we introduce a trainable weight tensor to indicate the importance of each node in different graphs. We can increase the performance of plenty of STGCNN models including [STGCN](https://www.ijcai.org/Proceedings/2018/0505), [ASTGCN](https://ojs.aaai.org/index.php/AAAI/article/view/3881), [MSTGCN](https://ojs.aaai.org/index.php/AAAI/article/view/3881), [ST-MGCN](https://ojs.aaai.org/index.php/AAAI/article/view/4247), and [Graph WaveNet](https://www.ijcai.org/Proceedings/2019/0264).

## Framework
![](https://github.com/swsamleo/HMSTGCN/blob/main/figures/framework.png)

## Datasets
### Histocical Records
- [Air Quality](https://english.mee.gov.cn/) The Ministry of Ecology and Environment of China (MEE) published a large-scale air quality dataset, comprising 92 air quality monitoring stations, to assess the hourly PM2.5 concentration in Jiangsu province in 2020.

### Weight Matrix
There are five weight matrices:
- Distance Graph: 
- Neighbor Graph: 
- Functionality graph:
- Heuristic Graph: 
- Temporal Pattern Similarity Graph:

### Multi-graph Spatial Embedding
The multi-graph spatial embedding is generated using [node2vec](https://github.com/aditya-grover/node2vec).


The historical records are stored in 'data\temporal_data', the weight matrix data is stores in 'data\graph', and the multi-graph spatial embedding data is stores in 'data\SE'.

## Requirements

Our code is based on Python3 (>= 3.6). The major libraries are listed as follows:
- torch >= 1.8.1
- numpy >= 1.15
- scipy >= 1.1.0
- torch-cluster >= 1.5.9
- torch-geometric >= 1.7.2
- torch-scatter >= 2.0.6
- torch-sparse >= 0.6.9
- torch-spline-conv >= 1.2.1
- pytorch-lightning >= 1.2.8
- wandb >= 0.11.1

The following command can help install the above libraries:
```powershell
pip install -r requirements.txt
```

## Usage
1. Generate training data
The time steps of historical observations and prediction horizons are both set to 24. The train, validation, and test part are divided into 7:1:2. Run the following command to generate training data:
```powershell
python generate_training_data.py
```
2. Parameters
- graphs: distance graph, neighbor graph, functionality graph, heuristic graph, temporal pattern similarity graph
- model: ASTGCN
- length of histotical observations: 24
- length of prediction horizons: 24
- epoches: 40
- batch size: 32
- learning rate: 1e-4
- weight decay: 1e-4
- number of attention heads: 8
- dimension of each attention outputs: 8

3. Train the model
Run the following command to train model:
```powershell
python train.py
```

## Citation
Please refer to our paper. Wei Shao*, Zhiling Jin*, Shuo Wang, et al, Flora Salim, Long-term Spatio-Temporal Forecasting via Dynamic Multiple-Graph Attention. [Long-term Spatio-Temporal Forecasting via Dynamic Multiple-Graph Attention](https://arxiv.org/abs/2204.11008?msclkid=3c019a95d0d611ec98d3ad7108897858). In Proceedings of the 31st International Joint Conference on Artificial Intelligence and the 25th European Conference on Artificial Intelligence (IJCAI-ECAI 2022), 2022.

```
@article{shao2022longterm,
  title={Long-term Spatio-temporal Forecasting via Dynamic Multiple-Graph Attention},
  author={Shao, Wei and Jin, Zhiling and Wang, Shuo and Kang, Yufan and Xiao, Xiao and Menouar, Hamid and Zhang, Zhaofeng and Zhang, Junshan and Salim, Flora},
  journal={arXiv preprint arXiv:2204.11008},
  year={2022}
}
```
