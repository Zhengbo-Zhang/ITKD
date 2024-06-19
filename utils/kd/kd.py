import torch
import torch.nn as nn
import torch.nn.functional as F

class KL_Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, stu_logit: torch.Tensor, tea_logit: torch.Tensor, T: float):
        res = self.kl(F.log_softmax(stu_logit/T, dim=1), F.softmax(tea_logit/T, dim=1))
        return res * T*T