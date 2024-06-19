import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from sklearn.utils import shuffle
from torch.distributions import Categorical

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x - mu) / (torch.exp(log_std) + 1e-8)).pow(2) + 2 * log_std + np.log(2 * np.pi))
    likelihood = pre_sum.sum(dim=1).view(-1, 1)
    return likelihood

class Actor(nn.Module):
    def __init__(self, state_dim, discrete_action_dim, t_action_dim, max_action, ):
        super(Actor, self).__init__()

        # self.l1_1 = nn.Linear(state_dim, 256)
        self.l1_2 = nn.Linear(state_dim, 256)

        # self.l2_1 = nn.Linear(256, 256)
        self.l2_2 = nn.Linear(256, 256)

        # self.l3_1 = nn.Linear(256, discrete_action_dim)
        self.l3_2 = nn.Linear(256, t_action_dim)

        self.max_action = max_action
        self.log_std = nn.Parameter(torch.zeros([10, t_action_dim]).view(-1, t_action_dim))
        self.p_log_std = nn.Parameter(torch.zeros([10, discrete_action_dim]).view(-1, discrete_action_dim))

    def forward(self, x):
        # 共享部分
        # x_1 = F.relu(self.l1_1(x))
        x_2 = F.relu(self.l1_2(x))

        # x_1 = F.relu(self.l2_1(x_1))
        x_2 = F.relu(self.l2_2(x_2))

        # 图片质量打分
        # q_mu = torch.tanh(self.l3_1(x_1))
        # q_log_std = self.p_log_std.sum(dim=0).view(1, -1) - 0.5
        # q_std = torch.exp(q_log_std)
        # q_noise = torch.FloatTensor(np.random.normal(
        #     0, 1, size=q_std.size())).type_as(q_std)
        # q_pi = q_mu + q_noise* q_std
        # q_action = 0 +(torch.sigmoid(q_pi) + 1) / 2 * 1  # 0~1
        # 蒸馏温度
        mu = torch.tanh(self.l3_2(x_2))
        log_std = self.log_std.sum(dim=0).view(1, -1) - 0.5
        std = torch.exp(log_std)
        return mu, std, log_std

        # return q_action, mu, std, log_std
    
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

