import torch
import torch.nn as nn


class DistriLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, num_groups=8, use_gpu=True):
        super(DistriLoss, self).__init__()
        self.num_classes = num_classes
        self.num_groups = num_groups
        self.use_gpu = use_gpu

        if self.use_gpu:
            # self.centers = nn.Parameter(torch.full((self.num_classes, self.num_groups), 0.5).cuda())
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.num_groups).cuda())
        else:
            # self.centers = nn.Parameter(torch.full((self.num_classes, self.num_groups), 0.5))
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.num_groups))

    def forward(self, group_weights, labels):
        """
        Args:
            group_weights: feature matrix with shape (batch_size, num_groups).
            labels: ground truth labels with shape (batch_size).
        """
        group_weights = group_weights.transpose(0, 1)
        batch_size = group_weights.size(0)
        loss = 0.0
        for i in range(batch_size):
            target_distri = self.centers[labels[i]]  # num_groups
            sample_distri = group_weights[i].squeeze()  # nunm_groups
            loss += torch.pow(target_distri - sample_distri, 2).sum()
        loss = loss / batch_size
        # print("Distribution loss: ", loss)
        return loss
