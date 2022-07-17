import torch
import torch.nn as nn
import torch.nn.functional as F


class CompactLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_groups (int): number of groups.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_groups=10, feat_dim=512, use_gpu=True):
        super(CompactLoss, self).__init__()
        self.num_groups = num_groups
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_groups, self.feat_dim).cuda())
            # self.centers = nn.Parameter(torch.full((self.num_groups, self.feat_dim), 0.5).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_groups, self.feat_dim))

    def forward(self, group_feats):
        """
        Args:
            group_feats: feature matrix with shape (num_groups, batch_size, feat_dim)
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        group_feats = group_feats.transpose(0, 1)  # (bs, num_groups, feat_dim)
        batch_size = group_feats.size(0)

        # norm
        centers = F.normalize(self.centers)

        loss = 0.0
        for i in range(batch_size):
            group_feat = group_feats[i]  # num_groups, feat_dim
            distmat = torch.pow(group_feat, 2).sum(1).unsqueeze(dim=1).expand(self.num_groups, self.num_groups) + \
                      torch.pow(centers, 2).sum(1).expand(self.num_groups, self.num_groups)
            distmat.addmm_(1, -2, group_feat, centers.t())

            distmat = distmat * (
                torch.eye(int(self.num_groups), int(self.num_groups)).cuda())  # 取对角线元素 对应group与其center的距离

            loss += distmat.clamp(min=1e-12, max=1e+12).sum() / self.num_groups

        loss = loss / batch_size
        # print("Compact loss: ", loss)
        return loss
