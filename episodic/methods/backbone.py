# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.utils import weight_norm


# --- gaussian initialize ---
def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
        L.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)


# --- LSTMCell module for matchingnet ---
class LSTMCell(nn.Module):
    maml = False

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        # if self.maml:
        #     self.x2h = Linear_fw(input_size, 4 * hidden_size, bias=bias)
        #     self.h2h = Linear_fw(hidden_size, 4 * hidden_size, bias=bias)
        # else:
        #     self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        #     self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)

        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden=None):
        if hidden is None:
            hx = torch.zeors_like(x)
            cx = torch.zeros_like(x)
        else:
            hx, cx = hidden

        gates = self.x2h(x) + self.h2h(hx)
        ingate, forgetgate, cellgate, outgate = torch.split(gates, self.hidden_size, dim=1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)
        hy = torch.mul(outgate, torch.tanh(cy))
        return (hy, cy)


# --- LSTM module for matchingnet ---
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, bidirectional=False):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.num_directions = 2 if bidirectional else 1
        assert (self.num_layers == 1)

        self.lstm = LSTMCell(input_size, hidden_size, self.bias)

    def forward(self, x, hidden=None):
        # swap axis if batch first
        if self.batch_first:
            x = x.permute(1, 0, 2)

        # hidden state
        if hidden is None:
            h0 = torch.zeros(self.num_directions, x.size(1), self.hidden_size, dtype=x.dtype, device=x.device)
            c0 = torch.zeros(self.num_directions, x.size(1), self.hidden_size, dtype=x.dtype, device=x.device)
        else:
            h0, c0 = hidden

        # forward
        outs = []
        hn = h0[0]
        cn = c0[0]
        for seq in range(x.size(0)):
            hn, cn = self.lstm(x[seq], (hn, cn))
            outs.append(hn.unsqueeze(0))
        outs = torch.cat(outs, dim=0)

        # reverse foward
        if self.num_directions == 2:
            outs_reverse = []
            hn = h0[1]
            cn = c0[1]
            for seq in range(x.size(0)):
                seq = x.size(1) - 1 - seq
                hn, cn = self.lstm(x[seq], (hn, cn))
                outs_reverse.append(hn.unsqueeze(0))
            outs_reverse = torch.cat(outs_reverse, dim=0)
            outs = torch.cat([outs, outs_reverse], dim=2)

        # swap axis if batch first
        if self.batch_first:
            outs = outs.permute(1, 0, 2)
        return outs


# --- flatten tensor ---
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


# --- softplus module ---
def softplus(x):
    return torch.nn.functional.softplus(x, beta=100)


# --- Simple ResNet Block ---
class SimpleBlock(nn.Module):
    # maml = False

    def __init__(self, indim, outdim, half_res, leaky=False):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        # if self.maml:
        #     self.C1 = Conv2d_fw(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
        #     self.BN1 = BatchNorm2d_fw(outdim)
        #     self.C2 = Conv2d_fw(outdim, outdim, kernel_size=3, padding=1, bias=False)
        #     self.BN2 = FeatureWiseTransformation2d_fw(
        #         outdim)  # feature-wise transformation at the end of each residual block
        # else:
        #     self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
        #     self.BN1 = nn.BatchNorm2d(outdim)
        #     self.C2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1, bias=False)
        #     self.BN2 = nn.BatchNorm2d(outdim)

        self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
        self.BN1 = nn.BatchNorm2d(outdim)
        self.C2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1, bias=False)
        self.BN2 = nn.BatchNorm2d(outdim)

        self.relu1 = nn.ReLU(inplace=True) if not leaky else nn.LeakyReLU(0.2, inplace=True)
        self.relu2 = nn.ReLU(inplace=True) if not leaky else nn.LeakyReLU(0.2, inplace=True)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim != outdim:
            # if self.maml:
            #     self.shortcut = Conv2d_fw(indim, outdim, 1, 2 if half_res else 1, bias=False)
            #     self.BNshortcut = FeatureWiseTransformation2d_fw(outdim)
            # else:
            #     self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
            #     self.BNshortcut = nn.BatchNorm2d(outdim)

            self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
            self.BNshortcut = nn.BatchNorm2d(outdim)

            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.BN2(out)
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
        out = out + short_out
        out = self.relu2(out)
        return out


#

# --- ResNet module ---
class ResNet(nn.Module):
    # maml = False

    def __init__(self, block, list_of_num_layers, list_of_out_dims, flatten=True, leakyrelu=False):
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNet, self).__init__()
        self.grads = []
        self.fmaps = []
        assert len(list_of_num_layers) == 4, 'Can have only four stages'

        conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        bn1 = nn.BatchNorm2d(64)

        relu = nn.ReLU(inplace=True) if not leakyrelu else nn.LeakyReLU(0.2, inplace=True)
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        init_layer(conv1)
        init_layer(bn1)

        trunk = [conv1, bn1, relu, pool1]

        indim = 64
        for i in range(4):
            for j in range(list_of_num_layers[i]):
                half_res = (i >= 1) and (j == 0)
                B = block(indim, list_of_out_dims[i], half_res, leaky=leakyrelu)
                trunk.append(B)
                indim = list_of_out_dims[i]

        if flatten:
            avgpool = nn.AvgPool2d(7)
            trunk.append(avgpool)
            trunk.append(Flatten())
            self.final_feat_dim = indim
        else:
            self.final_feat_dim = [indim, 7, 7]

        # add transformation layer
        # if n_dim == 128:
        #     trunk.append(TransformationLayer(self.final_feat_dim))

        self.trunk = nn.Sequential(*trunk)

    def forward(self, x):
        out = self.trunk(x)
        return out


class TransformationLayer(nn.Module):
    def __init__(self, n_dim):
        super(TransformationLayer, self).__init__()
        self.n_dim = n_dim
        self.trans_layer = nn.Sequential(
            nn.Linear(n_dim, 128)
        )

    def forward(self, x):
        return self.trans_layer(x)


# --- ResNet networks ---
def ResNet10(flatten=True, leakyrelu=False):
    return ResNet(SimpleBlock, [1, 1, 1, 1], [64, 128, 256, 512], flatten, leakyrelu)


def ResNet18(flatten=True, leakyrelu=False):
    return ResNet(SimpleBlock, [2, 2, 2, 2], [64, 128, 256, 512], flatten, leakyrelu)


model_dict = dict(ResNet10=ResNet10,
                  ResNet18=ResNet18)

# test usage
if __name__ == "__main__":
    model = ResNet18(n_dim=512)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(y.shape)
