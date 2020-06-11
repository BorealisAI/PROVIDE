# Copyright (c) 2019-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn

"""
An implementation of the 2D-LSTM cell. 
Gates equations are from 
https://arxiv.org/pdf/1810.03975.pdf

Each cell takes an input x, 2 hidden states and 2 cell states
In our case 2 hidden states are refinement and temporal. 
"""

class LSTM2dCell(nn.Module):

    # Initialized with x and hidden state dimentions
    def __init__(self, x_dim, h_dim):
        super(LSTM2dCell, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim

        # Initialize weight matrices for all 4 gates and the cell
        # For inputs
        self.W_x = nn.Linear(self.x_dim, self.h_dim * 5)
        # For temporal hidden states
        self.W_hor = nn.Linear(self.h_dim, self.h_dim * 5)
        # For refinement hidden states
        self.W_ver = nn.Linear(self.h_dim, self.h_dim * 5)

    def forward(self, x, h_temp, h_ref, c_temp, c_ref):

        gates_output = self.W_x(x) + self.W_hor(h_temp) + self.W_ver(h_ref)

        # input gate
        i = torch.sigmoid(gates_output[:, 0*self.h_dim:1*self.h_dim])
        # forget gate
        f = torch.sigmoid(gates_output[:, 1*self.h_dim:2*self.h_dim])
        # output gate
        o = torch.sigmoid(gates_output[:, 2*self.h_dim:3*self.h_dim])
        # cell
        c_tilda = torch.tanh(gates_output[:, 4*self.h_dim:])
        # lambda gate
        lmbd = torch.sigmoid(gates_output[:, 3*self.h_dim:4*self.h_dim])

        c = f * (lmbd * c_temp + (1 - lmbd) * c_ref) + c_tilda * i
        h = torch.tanh(c) * o

        # return a new hidden state h and a new cell state c
        return h, c