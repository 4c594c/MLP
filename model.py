'''
Author: 4c594c (aLong)
Date: 2022-12-09 13:51:49
LastEditors: aLong
LastEditTime: 2022-12-09 16:15:45
FilePath: \MLP\model.py
Description: 

Copyright (c) 2022 by 4c594c (aLong), All Rights Reserved. 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    """_summary_
    线性层,features是一维且要位于tensor最后一维度
    Args:
        nn (_type_): _description_
    """

    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        """_summary_

        Args:
            x (tensor): x维度[batchsize, in_features]

        Returns:
            tensor: 维度[batchsize, out_features]
        """
        return x @ self.weight + self.bias


class MLP(nn.Module):
    """_summary_
    多层感知机，输入层特征维度、隐藏层维度、输出层维度以及总层数(不包括输入层)
    Args:
        nn (_type_): _description_
    """

    def __init__(self,
                 in_features,
                 hidden_features,
                 out_features,
                 num_layers=2) -> None:
        super().__init__()
        self.in_linear = Linear(in_features, hidden_features)
        self.hidden_linears = nn.ModuleList([
            Linear(hidden_features, hidden_features)
            for i in range(num_layers - 2)
        ])
        self.out_linear = Linear(hidden_features, out_features)

        self.activations = nn.ModuleDict({
            'relu': nn.ReLU(),
            'lrelu': nn.LeakyReLU()
        })

    def forward(self, x: torch.tensor, act: str = 'relu'):
        activation = self.activations[act]
        x = activation(self.in_linear(x))
        for hidden_linear in self.hidden_linears:
            x = activation(hidden_linear(x))
        x = self.out_linear(x)
        return x


# for parameter in m.named_parameters():
#     print(parameter)

# for module in m.named_modules():
#     print(module)