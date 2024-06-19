import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from utils.kd.dkd import dkd_loss


class KL_Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.kl = nn.KLDivLoss(reduction="none")

    def forward(self, stu_logit: torch.Tensor, tea_logit: torch.Tensor, T):
        assert stu_logit.shape == tea_logit.shape, "logit shape should be same"
        # 判断T的类型
        if isinstance(T, torch.Tensor):
            T = T.reshape(-1, 1)
            assert (
                T.shape[0] == stu_logit.shape[0]
            ), "T shape should be same as stu_logit shape"

            # 分开算。因为torch 广播计算好像有问题，效果不好
            res = 0.0
            for i in range(stu_logit.shape[0]):
                res += (
                    F.kl_div(
                        F.log_softmax(stu_logit[i] / T[i], dim=-1),
                        F.softmax(tea_logit[i] / T[i], dim=-1),
                        reduction="sum",
                    )
                    * T[i]
                    * T[i]
                )
            return res / stu_logit.shape[0]

        elif isinstance(T, float):
            res = self.kl(
                F.log_softmax(stu_logit / T, dim=1), F.softmax(tea_logit / T, dim=1)
            )
            return res * T * T


class DKD_Loss(nn.Module):
    def __init__(self, alpha, beta, warmup=20) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.warmup = warmup

    def forward(
        self,
        stu_logit: torch.Tensor,
        tea_logit: torch.Tensor,
        gt: torch.Tensor,
        T,
        **kwargs,
    ):
        loss_dkd = min(kwargs["epoch"] / self.warmup, 1.0) * dkd_loss(
            stu_logit,
            tea_logit,
            gt,
            self.alpha,
            self.beta,
            T * 1.0,
        )
        return loss_dkd


class Instance_Temp_KD_Loss(nn.Module):
    """
    第一种： 只跟gt类做比较，其他类不参与计算
    第二种： predicted类和gt类都参与计算

    """

    def __init__(self, beta: float, loop_min: float, loop_max: float) -> None:
        super().__init__()
        self.beta = beta
        self.loop_min = loop_min
        self.loop_max = loop_max

        self.base_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self, stu_logit: torch.tensor, tea_logit: torch.Tensor, gt: torch.Tensor
    ):
        # get gt prob
        stu_softmax, tea_softmax = self.compute_probs(stu_logit, tea_logit)
        tea_pred = tea_softmax.gather(1, gt.unsqueeze(-1).long())
        tea_loss = torch.ones_like(tea_pred) - tea_pred

        discount_factor = (torch.cos(tea_loss * torch.pi) * self.beta + 1) / 2
        T = self.loop_min + discount_factor * (self.loop_max - self.loop_min)

        kd_loss = self.base_loss(
            F.log_softmax(stu_logit / T, dim=1), F.softmax(tea_logit / T, dim=1)
        )
        return kd_loss * T.mean() * T.mean(), T

    def compute_probs(self, stu_logit: torch.Tensor, tea_logit: torch.Tensor):
        return F.softmax(stu_logit, dim=1), F.softmax(tea_logit, dim=1)


class ITKD_Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, stu_logit: torch.Tensor, tea_logit: torch.Tensor, T: torch.Tensor
    ):
        # print(f'stu_logit: {stu_logit.shape}, tea_logit: {tea_logit.shape}, T: {T.shape}')
        T = T.reshape(-1, 1)
        kd_loss = F.kl_div(
            F.log_softmax(stu_logit / T, dim=1),
            F.softmax(tea_logit / T, dim=1),
            reduction="none",
        ).sum(dim=1)
        kd_loss = kd_loss * T * T
        return kd_loss.mean()


class VIDLoss(nn.Module):
    """Variational Information Distillation for Knowledge Transfer (CVPR 2019),
    code from author: https://github.com/ssahn0215/variational-information-distillation
    """

    def __init__(
        self,
        num_input_channels,
        num_mid_channel,
        num_target_channels,
        init_pred_var=5.0,
        eps=1e-5,
    ):
        super(VIDLoss, self).__init__()

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                bias=False,
                stride=stride,
            )

        self.regressor = nn.Sequential(
            conv1x1(num_input_channels, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_target_channels),
        )
        self.log_scale = torch.nn.Parameter(
            np.log(np.exp(init_pred_var - eps) - 1.0) * torch.ones(num_target_channels)
        )
        self.eps = eps

    def forward(self, input, target):
        # pool for dimentsion match
        s_H, t_H = input.shape[2], target.shape[2]
        if s_H > t_H:
            input = F.adaptive_avg_pool2d(input, (t_H, t_H))
        elif s_H < t_H:
            target = F.adaptive_avg_pool2d(target, (s_H, s_H))
        else:
            pass
        pred_mean = self.regressor(input)
        pred_var = torch.log(1.0 + torch.exp(self.log_scale)) + self.eps
        pred_var = pred_var.view(1, -1, 1, 1)
        neg_log_prob = 0.5 * (
            (pred_mean - target) ** 2 / pred_var + torch.log(pred_var)
        )
        loss = torch.mean(neg_log_prob)
        return loss


class Similarity(nn.Module):
    """Similarity-Preserving Knowledge Distillation, ICCV2019, verified by original author"""

    def __init__(self):
        super(Similarity, self).__init__()

    def forward(self, g_s, g_t):
        return [self.similarity_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]

    def similarity_loss(self, f_s, f_t):
        bsz = f_s.shape[0]
        f_s = f_s.view(bsz, -1)
        f_t = f_t.view(bsz, -1)

        G_s = torch.mm(f_s, torch.t(f_s))
        # G_s = G_s / G_s.norm(2)
        G_s = torch.nn.functional.normalize(G_s, dim=1)
        G_t = torch.mm(f_t, torch.t(f_t))
        # G_t = G_t / G_t.norm(2)
        G_t = torch.nn.functional.normalize(G_t, dim=1)

        G_diff = G_t - G_s
        loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
        return loss


class Normalize(nn.Module):
    """normalization layer"""

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1.0 / self.power)
        out = x.div(norm)
        return out


class ContrastLoss(nn.Module):
    """
    contrastive loss, corresponding to Eq (18)
    """

    def __init__(self, n_data):
        super(ContrastLoss, self).__init__()
        self.n_data = n_data

    def forward(self, x):
        bsz = x.shape[0]
        m = x.size(1) - 1

        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()

        # loss for K negative pair
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()

        loss = -(log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        return loss


class AliasMethod(object):
    """
    From: https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    """

    def __init__(self, probs):
        if probs.sum() > 1:
            probs.div_(probs.sum())
        K = len(probs)
        self.prob = torch.zeros(K)
        self.alias = torch.LongTensor([0] * K)

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = K * prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.alias[small] = large
            self.prob[large] = (self.prob[large] - 1.0) + self.prob[small]

            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller + larger:
            self.prob[last_one] = 1

    def cuda(self):
        self.prob = self.prob.cuda()
        self.alias = self.alias.cuda()

    def draw(self, N):
        """Draw N samples from multinomial"""
        K = self.alias.size(0)

        kk = torch.zeros(N, dtype=torch.long, device=self.prob.device).random_(0, K)
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        # b is whether a random number is greater than q
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1 - b).long())

        return oq + oj


class ContrastMemory(nn.Module):
    """
    memory buffer that supplies large amount of negative samples.
    """

    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5):
        super(ContrastMemory, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K

        self.register_buffer("params", torch.tensor([K, T, -1, -1, momentum]))
        stdv = 1.0 / math.sqrt(inputSize / 3)
        self.register_buffer(
            "memory_v1", torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv)
        )
        self.register_buffer(
            "memory_v2", torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv)
        )

    def forward(self, v1, v2, y, idx=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_v1 = self.params[2].item()
        Z_v2 = self.params[3].item()

        momentum = self.params[4].item()
        batchSize = v1.size(0)
        outputSize = self.memory_v1.size(0)
        inputSize = self.memory_v1.size(1)

        # original score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)
        # sample
        weight_v1 = torch.index_select(self.memory_v1, 0, idx.view(-1)).detach()
        weight_v1 = weight_v1.view(batchSize, K + 1, inputSize)
        out_v2 = torch.bmm(weight_v1, v2.view(batchSize, inputSize, 1))
        out_v2 = torch.exp(torch.div(out_v2, T))
        # sample
        weight_v2 = torch.index_select(self.memory_v2, 0, idx.view(-1)).detach()
        weight_v2 = weight_v2.view(batchSize, K + 1, inputSize)
        out_v1 = torch.bmm(weight_v2, v1.view(batchSize, inputSize, 1))
        out_v1 = torch.exp(torch.div(out_v1, T))

        # set Z if haven't been set yet
        if Z_v1 < 0:
            self.params[2] = out_v1.mean() * outputSize
            Z_v1 = self.params[2].clone().detach().item()
            print("normalization constant Z_v1 is set to {:.1f}".format(Z_v1))
        if Z_v2 < 0:
            self.params[3] = out_v2.mean() * outputSize
            Z_v2 = self.params[3].clone().detach().item()
            print("normalization constant Z_v2 is set to {:.1f}".format(Z_v2))

        # compute out_v1, out_v2
        out_v1 = torch.div(out_v1, Z_v1).contiguous()
        out_v2 = torch.div(out_v2, Z_v2).contiguous()

        # update memory
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_v1, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(v1, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v1 = l_pos.div(l_norm)
            self.memory_v1.index_copy_(0, y, updated_v1)

            ab_pos = torch.index_select(self.memory_v2, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(v2, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v2 = ab_pos.div(ab_norm)
            self.memory_v2.index_copy_(0, y, updated_v2)

        return out_v1, out_v2


class CRDLoss(nn.Module):
    """CRD Loss function
    includes two symmetric parts:
    (a) using teacher as anchor, choose positive and negatives over the student side
    (b) using student as anchor, choose positive and negatives over the teacher side

    Args:
        opt.s_dim: the dimension of student's feature
        opt.t_dim: the dimension of teacher's feature
        opt.feat_dim: the dimension of the projection space
        opt.nce_k: number of negatives paired with each positive
        opt.nce_t: the temperature
        opt.nce_m: the momentum for updating the memory buffer
        opt.n_data: the number of samples in the training set, therefor the memory buffer is: opt.n_data x opt.feat_dim
    """

    def __init__(self, opt):
        super(CRDLoss, self).__init__()
        self.embed_s = Embed(opt.s_dim, opt.feat_dim)
        self.embed_t = Embed(opt.t_dim, opt.feat_dim)
        self.contrast = ContrastMemory(
            opt.feat_dim, opt.n_data, opt.nce_k, opt.nce_t, opt.nce_m
        )
        self.criterion_t = ContrastLoss(opt.n_data)
        self.criterion_s = ContrastLoss(opt.n_data)

    def forward(self, f_s, f_t, idx, contrast_idx=None):
        """
        Args:
            f_s: the feature of student network, size [batch_size, s_dim]
            f_t: the feature of teacher network, size [batch_size, t_dim]
            idx: the indices of these positive samples in the dataset, size [batch_size]
            contrast_idx: the indices of negative samples, size [batch_size, nce_k]

        Returns:
            The contrastive loss
        """
        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)
        out_s, out_t = self.contrast(f_s, f_t, idx, contrast_idx)
        s_loss = self.criterion_s(out_s)
        t_loss = self.criterion_t(out_t)
        loss = s_loss + t_loss
        return loss


class ContrastMemory(nn.Module):
    """
    memory buffer that supplies large amount of negative samples.
    """

    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5):
        super(ContrastMemory, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K

        self.register_buffer("params", torch.tensor([K, T, -1, -1, momentum]))
        stdv = 1.0 / math.sqrt(inputSize / 3)
        self.register_buffer(
            "memory_v1", torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv)
        )
        self.register_buffer(
            "memory_v2", torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv)
        )

    def forward(self, v1, v2, y, idx=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_v1 = self.params[2].item()
        Z_v2 = self.params[3].item()

        momentum = self.params[4].item()
        batchSize = v1.size(0)
        outputSize = self.memory_v1.size(0)
        inputSize = self.memory_v1.size(1)

        # original score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)
        # sample
        weight_v1 = torch.index_select(self.memory_v1, 0, idx.view(-1)).detach()
        weight_v1 = weight_v1.view(batchSize, K + 1, inputSize)
        out_v2 = torch.bmm(weight_v1, v2.view(batchSize, inputSize, 1))
        out_v2 = torch.exp(torch.div(out_v2, T))
        # sample
        weight_v2 = torch.index_select(self.memory_v2, 0, idx.view(-1)).detach()
        weight_v2 = weight_v2.view(batchSize, K + 1, inputSize)
        out_v1 = torch.bmm(weight_v2, v1.view(batchSize, inputSize, 1))
        out_v1 = torch.exp(torch.div(out_v1, T))

        # set Z if haven't been set yet
        if Z_v1 < 0:
            self.params[2] = out_v1.mean() * outputSize
            Z_v1 = self.params[2].clone().detach().item()
            print("normalization constant Z_v1 is set to {:.1f}".format(Z_v1))
        if Z_v2 < 0:
            self.params[3] = out_v2.mean() * outputSize
            Z_v2 = self.params[3].clone().detach().item()
            print("normalization constant Z_v2 is set to {:.1f}".format(Z_v2))

        # compute out_v1, out_v2
        out_v1 = torch.div(out_v1, Z_v1).contiguous()
        out_v2 = torch.div(out_v2, Z_v2).contiguous()

        # update memory
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_v1, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(v1, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v1 = l_pos.div(l_norm)
            self.memory_v1.index_copy_(0, y, updated_v1)

            ab_pos = torch.index_select(self.memory_v2, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(v2, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v2 = ab_pos.div(ab_norm)
            self.memory_v2.index_copy_(0, y, updated_v2)

        return out_v1, out_v2


class AliasMethod(object):
    """
    From: https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    """

    def __init__(self, probs):
        if probs.sum() > 1:
            probs.div_(probs.sum())
        K = len(probs)
        self.prob = torch.zeros(K)
        self.alias = torch.LongTensor([0] * K)

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = K * prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.alias[small] = large
            self.prob[large] = (self.prob[large] - 1.0) + self.prob[small]

            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller + larger:
            self.prob[last_one] = 1

    def cuda(self):
        self.prob = self.prob.cuda()
        self.alias = self.alias.cuda()

    def draw(self, N):
        """Draw N samples from multinomial"""
        K = self.alias.size(0)

        kk = torch.zeros(N, dtype=torch.long, device=self.prob.device).random_(0, K)
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        # b is whether a random number is greater than q
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1 - b).long())

        return oq + oj


eps = 1e-7


class CRDLoss(nn.Module):
    """CRD Loss function
    includes two symmetric parts:
    (a) using teacher as anchor, choose positive and negatives over the student side
    (b) using student as anchor, choose positive and negatives over the teacher side

    Args:
        opt.s_dim: the dimension of student's feature
        opt.t_dim: the dimension of teacher's feature
        opt.feat_dim: the dimension of the projection space
        opt.nce_k: number of negatives paired with each positive
        opt.nce_t: the temperature
        opt.nce_m: the momentum for updating the memory buffer
        opt.n_data: the number of samples in the training set, therefor the memory buffer is: opt.n_data x opt.feat_dim
    """

    def __init__(self, opt):
        super(CRDLoss, self).__init__()
        self.embed_s = Embed(opt.s_dim, opt.feat_dim)
        self.embed_t = Embed(opt.t_dim, opt.feat_dim)
        self.contrast = ContrastMemory(
            opt.feat_dim, opt.n_data, opt.nce_k, opt.nce_t, opt.nce_m
        )
        self.criterion_t = ContrastLoss(opt.n_data)
        self.criterion_s = ContrastLoss(opt.n_data)

    def forward(self, f_s, f_t, idx, contrast_idx=None):
        """
        Args:
            f_s: the feature of student network, size [batch_size, s_dim]
            f_t: the feature of teacher network, size [batch_size, t_dim]
            idx: the indices of these positive samples in the dataset, size [batch_size]
            contrast_idx: the indices of negative samples, size [batch_size, nce_k]

        Returns:
            The contrastive loss
        """
        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)
        out_s, out_t = self.contrast(f_s, f_t, idx, contrast_idx)
        s_loss = self.criterion_s(out_s)
        t_loss = self.criterion_t(out_t)
        loss = s_loss + t_loss
        return loss


class ContrastLoss(nn.Module):
    """
    contrastive loss, corresponding to Eq (18)
    """

    def __init__(self, n_data):
        super(ContrastLoss, self).__init__()
        self.n_data = n_data

    def forward(self, x):
        bsz = x.shape[0]
        m = x.size(1) - 1

        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()

        # loss for K negative pair
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()

        loss = -(log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        return loss


class Embed(nn.Module):
    """Embedding module"""

    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x


class VIDLoss(nn.Module):
    """Variational Information Distillation for Knowledge Transfer (CVPR 2019),
    code from author: https://github.com/ssahn0215/variational-information-distillation
    """

    def __init__(
        self,
        num_input_channels,
        num_mid_channel,
        num_target_channels,
        init_pred_var=5.0,
        eps=1e-5,
    ):
        super(VIDLoss, self).__init__()

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                bias=False,
                stride=stride,
            )

        self.regressor = nn.Sequential(
            conv1x1(num_input_channels, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_target_channels),
        )
        self.log_scale = torch.nn.Parameter(
            np.log(np.exp(init_pred_var - eps) - 1.0) * torch.ones(num_target_channels)
        )
        self.eps = eps

    def forward(self, input, target):
        # pool for dimentsion match
        s_H, t_H = input.shape[2], target.shape[2]
        if s_H > t_H:
            input = F.adaptive_avg_pool2d(input, (t_H, t_H))
        elif s_H < t_H:
            target = F.adaptive_avg_pool2d(target, (s_H, s_H))
        else:
            pass
        pred_mean = self.regressor(input)
        pred_var = torch.log(1.0 + torch.exp(self.log_scale)) + self.eps
        pred_var = pred_var.view(1, -1, 1, 1)
        neg_log_prob = 0.5 * (
            (pred_mean - target) ** 2 / pred_var + torch.log(pred_var)
        )
        loss = torch.mean(neg_log_prob)
        return loss


class PKT(nn.Module):
    """Probabilistic Knowledge Transfer for deep representation learning
    Code from author: https://github.com/passalis/probabilistic_kt"""

    def __init__(self):
        super(PKT, self).__init__()

    def forward(self, f_s, f_t):
        return self.cosine_similarity_loss(f_s, f_t)

    @staticmethod
    def cosine_similarity_loss(output_net, target_net, eps=0.0000001):
        # Normalize each vector by its norm
        output_net_norm = torch.sqrt(torch.sum(output_net**2, dim=1, keepdim=True))
        output_net = output_net / (output_net_norm + eps)
        output_net[output_net != output_net] = 0

        target_net_norm = torch.sqrt(torch.sum(target_net**2, dim=1, keepdim=True))
        target_net = target_net / (target_net_norm + eps)
        target_net[target_net != target_net] = 0

        # Calculate the cosine similarity
        model_similarity = torch.mm(output_net, output_net.transpose(0, 1))
        target_similarity = torch.mm(target_net, target_net.transpose(0, 1))

        # Scale cosine similarity to 0..1
        model_similarity = (model_similarity + 1.0) / 2.0
        target_similarity = (target_similarity + 1.0) / 2.0

        # Transform them into probabilities
        model_similarity = model_similarity / torch.sum(
            model_similarity, dim=1, keepdim=True
        )
        target_similarity = target_similarity / torch.sum(
            target_similarity, dim=1, keepdim=True
        )

        # Calculate the KL-divergence
        loss = torch.mean(
            target_similarity
            * torch.log((target_similarity + eps) / (model_similarity + eps))
        )

        return loss


class SRRL(nn.Module):
    """ICLR-2021: Knowledge Distillation via Softmax Regression Representation Learning"""

    def __init__(self, *, s_n, t_n):
        super(SRRL, self).__init__()

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                stride=stride,
                bias=False,
            )

        setattr(
            self,
            "transfer",
            nn.Sequential(
                conv1x1(s_n, t_n),
                nn.BatchNorm2d(t_n),
                nn.ReLU(inplace=True),
            ),
        )

    def forward(self, feat_s, cls_t):
        feat_s = feat_s.unsqueeze(-1).unsqueeze(-1)
        temp_feat = self.transfer(feat_s)
        trans_feat_s = temp_feat.view(temp_feat.size(0), -1)
        pred_feat_s = cls_t(trans_feat_s)
        return trans_feat_s, pred_feat_s
