import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Mine(nn.Module):
    def __init__(self, input_size=512 * 2, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, input):
        output = F.leaky_relu(self.fc1(input))
        output = F.leaky_relu(self.fc2(output))
        return output
