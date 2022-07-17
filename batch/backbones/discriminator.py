from torch import nn


class Discriminator(nn.Module):
    def __init__(self, in_dim, n_domains):
        super(Discriminator, self).__init__()
        self.d = 64
        lk = 0.01

        self.fc = nn.Sequential(
            nn.Linear(in_dim, in_dim // 4, bias=False),
            nn.Dropout(),
            nn.BatchNorm1d(in_dim // 4, affine=True),
            nn.LeakyReLU(lk, inplace=True),

            nn.Linear(in_dim // 4, n_domains, bias=False),
            nn.BatchNorm1d(n_domains, affine=True),
            #
            # nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)
