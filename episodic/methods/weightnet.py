from torch import nn


class WeightNet(nn.Module):
    def __init__(self, in_dim):
        super(WeightNet, self).__init__()

        self.compute_weight = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Dropout(p=0.5),
            nn.ReLU(),

            nn.Linear(in_dim // 2, in_dim // 8),
            nn.Dropout(p=0.5),
            nn.ReLU(),

            nn.Linear(in_dim // 8, 1)
        )
        # self.compute_weight = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.compute_weight(x)
