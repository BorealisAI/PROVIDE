# Copyright (c) 2019-present, Royal Bank of Canada.
# Copyright (c) 2019-present, Michael Kelly.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

####################################################################################
# Code is based on the IODINE (https://arxiv.org/pdf/1903.00450.pdf) implementation 
# from https://github.com/MichaelKevinKelly/IODINE by Michael Kelly
####################################################################################

import torch
from src.networks.lstm2d_cell import LSTM2dCell

"""
Implementation of the refinement network for the model. The main different from the IODINE is a use of the 2D-LSTM cell.
"""
class RefineNetLSTM(torch.nn.Module):

	def __init__(self, z_dim, channels_in):
		super(RefineNetLSTM, self).__init__()
		self.convnet = ConvNet(channels_in)
		self.fc_in = torch.nn.Sequential(torch.nn.Linear(16384,128),torch.nn.ELU())
		# Create a 2D-LSTM
		self.lstm = LSTM2dCell(128+4*z_dim, 128)
		self.fc_out = torch.nn.Linear(128,2*z_dim)

	def forward(self, x, h_1, c_1, h_2, c_2):
		x_img, x_vec = x['img'], x['vec']
		N,C,H,W = x_img.shape
		conv_codes = self.convnet(x_img)
		conv_codes = self.fc_in(conv_codes.view(N,-1))
		lstm_input = torch.cat((x_vec,conv_codes),dim=1)
		# 2D-LSTM takes as inputs two hidden states (temporal and refinement) and two cell states
		h,c = self.lstm(lstm_input, h_1, h_2, c_1, c_2)
		return self.fc_out(h), h, c

class ConvNet(torch.nn.Module):
	
	def __init__(self, channels_in):
		super(ConvNet, self).__init__()
	
		self.model = torch.nn.Sequential(
			torch.nn.Conv2d(channels_in,64,kernel_size=3,stride=1,padding=1),
			torch.nn.ELU(),
			torch.nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
			torch.nn.ELU(),
			torch.nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
			torch.nn.ELU(),
			torch.nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
			torch.nn.ELU(),
			torch.nn.AvgPool2d(4))

	def forward(self, x):
		return self.model(x)
