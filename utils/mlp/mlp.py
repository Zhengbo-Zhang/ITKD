from torch.autograd import Function
from typing import Callable, Union
from torch import Tensor
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module, _grad_t
from torch.utils.hooks import RemovableHandle

from utils.kd.dkd import mlp_dkd_loss


'''
随便加个attention

'''


class MLP_T(nn.Module):
    def __init__(self, in_chan: int, ) -> None:
        super().__init__()
        self.token = nn.Linear(in_chan, 128)
        self.qkv = nn.Linear(128, 384)
        self.fc_out = nn.Linear(128, 1)
        self.attn_drop = nn.Dropout(0.1)

        self.grl = GradientReversal()
        
        # self.register_full_backward_hook(self.backward_hook)

    def forward(self, x, decay_value):
        x = self.token(x)
        qkv = self.qkv(x)
        q, k, v = torch.split(qkv, 128, dim=1)

        att_scores = q @ k.T
        att_scores = torch.sigmoid(att_scores)
        att_scores = self.attn_drop(att_scores)
        att = att_scores @ v

        out = self.fc_out(att)

        T = self.grl(out, decay_value)

        return T

    def backward_hook(self, module, grad_in, grad_out):
        # print(f'module={module}, grad_in={grad_in}, grad_out={grad_out}')
        out = list(grad_in)

        for i, grad in enumerate(grad_in):
            out[i] = -grad

        # print(f'out={out}')

        return tuple(out)


class MLP_Loss(nn.Module):
    def __init__(self, in_chan: int) -> None:
        super().__init__()

        self.T_network = MLP_T(in_chan)

    def forward(self, tea_logit: torch.Tensor, stu_logit: torch.Tensor, gt: torch.Tensor, device, decay_value):

        if len(gt.shape) == 1:
            gt = gt.unsqueeze(1)
        tea_prob = F.softmax(tea_logit, dim=1)
        stu_prob = F.softmax(stu_logit, dim=1)

        x = torch.cat([tea_prob, stu_prob], dim=1)

        '''
        mlp network
        '''
        self.T_network = self.T_network.to(device)
        T = self.T_network(x, decay_value)

        '''
        loss
        '''
        T = T.reshape(-1, 1)

        T = 1 + (torch.tanh(T) + 1) / 2 * (10-1)

        kd_loss = F.kl_div(F.log_softmax(stu_logit/T, dim=1),
                           F.softmax(tea_logit/T, dim=1), reduction='none').sum(dim=1)
        kd_loss = kd_loss*T*T
        return kd_loss.mean(), T
    
class MLPDKD_Loss(MLP_Loss):
    def __init__(self, in_chan: int, alpha: int=1, beta:int=8) -> None:
        super().__init__(in_chan)
        self.alpha=alpha
        self.beta=beta
        
        
    def forward(self, tea_logit: Tensor, stu_logit: Tensor, gt: Tensor, device, decay_value):
        if len(gt.shape) == 1:
            gt = gt.unsqueeze(1)
        tea_prob = F.softmax(tea_logit, dim=1)
        stu_prob = F.softmax(stu_logit, dim=1)

        x = torch.cat([tea_prob, stu_prob], dim=1)

        '''
        mlp network
        '''
        self.T_network = self.T_network.to(device)
        T = self.T_network(x, decay_value)

        '''
        loss
        '''
        T = T.reshape(-1, 1)

        T = 1 + (torch.tanh(T) + 1) / 2 * (10-1)
        loss_dkd = mlp_dkd_loss(
            stu_logit,
            tea_logit,
            gt,
            self.alpha,
            self.beta,
            T
        )
        return loss_dkd, T

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self):
        super(GradientReversal, self).__init__()
        # self.lambda_ = lambda_

    def forward(self, x, lambda_):
        return GradientReversalFunction.apply(x, lambda_)
