from torch import nn
import torch
import torch.nn.functional as F


class WeightNet(nn.Module):
    def __init__(self):
        super(WeightNet, self).__init__()
        self.scale = nn.Parameter(torch.tensor(5.))

    def forward(self, x, p):
        logits = torch.mm(F.normalize(x, dim=-1),
                          F.normalize(p, dim=-1).t())
        sim = torch.diagonal(logits)
        return sim * self.scale
