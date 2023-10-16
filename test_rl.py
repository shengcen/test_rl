import numpy as np

import os
import sys

import random
import mpnn as gnn
import tensorflow as tf

import torch
import datetime
import time
import math
from mpnn import GCN, GCN_test
from sklearn.metrics import mean_squared_error
import torch.nn as nn


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'




class High_agent:
    def __init__(self, batch_size):
        # self.memory = deque(maxlen=MAX_QUEUE_SIZE)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        # self.epsilon_decay = 0.995
        self.epsilon_decay = 0.985
        self.primary_network = GCN_test()
        self.target_network = GCN_test()
        self.optimizer_primary = torch.optim.Adam(self.primary_network.parameters(), lr=1000)
        self.optimizer_target = torch.optim.Adam(self.target_network.parameters(), lr=1000)
        self.numbersamples = batch_size
        self.listQValues = None
        # self.numbersamples_high = 32
        self.numbersamples_high = 24
        self.criterion = torch.nn.MSELoss()


    def replay(self, epi):

        left_list = []
        right_list = []

        x1 = torch.Tensor([2,3,4,5])
        x2 = torch.Tensor([4,6,2,7])
        x3 = torch.Tensor([6,3,1,6])
        x4 = torch.Tensor([5,3,6,3])
        r1 = 3
        r2 = 8

        qsa1 = self.primary_network(x1)[0][1]
        qsa2 = self.primary_network(x2)[0][0]

        aa1 = self.target_network(x3).detach()
        aa2 = self.target_network(x4).detach()
        q_next_max1 = torch.max(aa1[0])
        q_next_max2 = torch.max(aa2[0])

        left_list.append(qsa1)
        left_list.append(qsa2)
        right_list.append(r1 + self.gamma * q_next_max1)
        right_list.append(r2 + self.gamma * q_next_max2)

        left_list = torch.tensor(left_list, dtype=torch.float32, requires_grad=True)
        right_list = torch.tensor(right_list, dtype=torch.float32)

        # print("left: ")
        # print(left_list)
        # print("right: ")
        # print(right_list)

        loss = self.criterion(left_list, right_list)

        self.optimizer_primary.zero_grad()
        loss.backward()
        self.optimizer_primary.step()




high_agent = High_agent(16)

for ep_it in range(10):
    high_agent.replay(ep_it)
    x = torch.Tensor([5,3,5,9])
    print(high_agent.primary_network(x))