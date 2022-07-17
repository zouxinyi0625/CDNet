from __future__ import print_function
from methods.transferLearning_clfHeads import softMax, cosMax, arcMax
from torch.autograd import Variable
from torch.optim import Adam
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from backbones.shallow_backbone import Conv4Net, Flatten
from backbones.utils import device_kwargs
from backbones.weightnet import WeightNet
from backbones.discriminator import Discriminator
from utils import compute_entropy
import torch.nn.functional as F
from losses.compact_loss import CompactLoss


def clf_fun(self, n_class, device, s=20, m=0.01):
    if self.method == 'softMax':
        clf = softMax(self.out_dim, n_class).to(device)
    elif self.method == 'cosMax':
        clf = cosMax(self.out_dim, n_class, s).to(device)
    elif self.method == 'arcMax':
        clf = arcMax(self.out_dim, n_class, s, m).to(device)

    return clf


class transferLearningFuns(nn.Module):
    def __init__(self, args, net, n_class, writer):
        super(transferLearningFuns, self).__init__()

        self.device = device_kwargs(args)
        self.method = args.method
        self.lr = args.lr
        self.backbone = args.backbone
        self.args = args

        self.n_epoch = args.n_epoch
        self.n_class = n_class
        self.n_support = args.n_shot
        self.n_query = args.n_query

        self.out_dim = args.out_dim
        self.lr = args.lr

        # encoder
        self.net = net.to(self.device)
        self.over_fineTune = args.over_fineTune

        # decomposition
        self.colors = args.color
        self.feat_dim = 512

        self.decomposition = nn.Sequential(
            nn.Linear(self.out_dim, self.feat_dim),
            nn.BatchNorm1d(self.feat_dim),
            nn.PReLU()
        )

        # weightnet
        self.weight_net = WeightNet(self.out_dim + self.feat_dim)

        # discriminator
        self.discriminator = Discriminator(self.feat_dim, args.n_domains)

        # classifier
        self.base_clf = clf_fun(self, self.n_class, self.device)

        self.ft_n_epoch = args.ft_n_epoch
        self.n_way = args.n_way

        self.writer = writer

        # self.CLoss = CompactLoss(num_groups=self.colors)

        # fix encoder
        self.optimizer = Adam([{'params': self.decomposition.parameters()},
                               {'params': self.discriminator.parameters()},
                               {'params': self.weight_net.parameters()},
                               {'params': self.base_clf.parameters()}],
                              lr=1e-4)

    def accuracy_fun_tl(self, data_loader):
        Acc = 0
        self.net.eval()
        with torch.no_grad():
            for x, y in data_loader:
                x, y = Variable(x).to(self.device), Variable(y).to(self.device)
                logits = self.clf(self.net(x))
                y_hat = np.argmax(logits.data.cpu().numpy(), axis=1)
                Acc += np.mean((y_hat == y.data.cpu().numpy()).astype(int))
        return Acc.item() / len(data_loader)

    def acc_feature(self, logits, y):
        y_hat = np.argmax(logits.data.cpu().numpy(), axis=1)
        Acc = np.mean((y_hat == y.data.cpu().numpy()).astype(int))
        return Acc.item()

    def accuracy_fun(self, x, n_way):
        # novel_clf = clf_fun(self, self.n_way, self.device, s=5)
        # novel_optimizer = torch.optim.Adam(novel_clf.parameters(), lr=self.lr)
        x_support = x[:, :self.n_support, :, :, :].contiguous()
        x_support = x_support.view(n_way * self.n_support, *x.size()[2:])
        # y_support = torch.from_numpy(np.repeat(range(n_way), self.n_support))
        # y_support = Variable(y_support.to(self.device))

        x_query = x[:, self.n_support:, :, :, :].contiguous()
        x_query = x_query.view(n_way * self.n_query, *x.size()[2:])
        y_query = torch.from_numpy(np.repeat(range(n_way), self.n_query))
        y_query = Variable(y_query.cuda())

        with torch.no_grad():
            z_support, _, _, _, _ = self.extract_feature(x_support)
            z_query, _, _, _, _ = self.extract_feature(x_query)

        # proto classifier
        z_proto = z_support.view(self.n_way, self.n_support, -1).mean(1)

        dists = self.euclidean_dist(z_query, z_proto)
        scores = -dists

        y_hat = np.argmax(scores.data.cpu().numpy(), axis=1)

        return np.mean((y_hat == y_query.data.cpu().numpy()).astype(int)) * 100

    def euclidean_dist(self, x, y):
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        return torch.pow(x - y, 2).sum(2)


    def extract_feature(self, x):
        x = self.net(x)  # ori feature
        ori_x = x

        re_x = 0.0

        primarys = []
        weights = []

        for i in range(self.colors):
            # decompostion
            primary = self.decomposition(x)
            # primary = primary.mean(dim=0)  # 取当前batch的均值
            input_f = torch.cat([x, primary.expand(x.shape[0], self.out_dim)], dim=1)
            weight = self.weight_net(input_f)

            primarys.append(primary)
            weights.append(weight)

            x = x - primary * weight

            # reconstruct
            re_x += primary * weight

        primarys = torch.stack(primarys)  # [color, b, dim]
        weights = torch.stack(weights)  # [color,batch,1]

        return re_x, x, ori_x, primarys, weights

    # note: this is typical/batch based training
    def train_loop(self, x, y, y_domain, epoch):
        self.net.train()
        self.decomposition.train()
        self.weight_net.train()
        self.base_clf.train()
        self.discriminator.train()
        loss_sum = 0

        # for i, (x, y) in enumerate(trainLoader):
        x, y = Variable(x).to(self.device), Variable(y).to(self.device)

        # forward
        # re_x 重构的特征, de_x 分解后的特征, ori_x embedding原始特征, primarys 分解出的原色, weights 得到的权重
        re_x, de_x, ori_x, primarys, weights = self.extract_feature(x)

        # 重构后的特征用于分类
        loss_ce = self.base_clf.loss(self.base_clf(re_x), y)

        # 分解后的特征不包含情感信息
        # logits_de = self.base_clf(de_x)
        # loss_de = -compute_entropy(F.softmax(logits_de))

        # 重构与原始特征之差 (用logits的KL divergence?)
        # loss_t = nn.MSELoss()(re_x, ori_x)

        # 域信息判断
        y_domains = torch.tensor([y_domain] * len(x)).cuda()
        loss_domain = nn.CrossEntropyLoss()(self.discriminator(de_x), y_domains)

        # primary compact loss
        # loss_compact = self.CLoss(primarys)

        # 原色不包含域信息
        # logits_p_domain = self.discriminator(primarys.reshape(-1, self.feat_dim))
        # loss_pd = -compute_entropy(F.softmax(logits_p_domain))

        # loss = self.args.w_ce * loss_ce + self.args.w_de * loss_de + self.args.w_t * loss_t + self.args.w_domain * loss_domain + self.args.w_c * loss_compact + self.args.w_pd *loss_pd
        loss = self.args.w_ce * loss_ce + self.args.w_domain * loss_domain
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss_sum += loss.item()

        self.writer.add_scalar("training" + '/ce_loss', loss_ce.item(), epoch)
        # self.writer.add_scalar("training" + '/de_loss', loss_de.item(), epoch)
        # self.writer.add_scalar("training" + '/t_loss', loss_t.item(), epoch)
        self.writer.add_scalar("training" + '/domain_loss', loss_domain.item(), epoch)
        # self.writer.add_scalar("training" + '/compact_loss', loss_compact.item(), epoch)
        # self.writer.add_scalar("training" + '/p_domain_loss', loss_pd.item(), epoch)

        # print("Iter{}, loss: {}".format(epoch, loss.item()))

        return loss_sum

    def test_fer(self, testLoader):
        self.net.eval()

        correct = 0
        total = 0

        for i, (x, y) in enumerate(testLoader):
            x, y = Variable(x).to(self.device), Variable(y).to(self.device)
            scores = self.base_clf(self.net(x))
            pred = scores.max(1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()
            total += len(y)

        return (correct / total) * 100.0

    # note this is episodic testing
    def test_loop(self, test_loader, n_way, epoch):
        acc_all = []
        self.net.eval()
        self.decomposition.eval()
        self.weight_net.eval()
        self.discriminator.eval()
        for i, (x, _) in enumerate(test_loader):
            x = Variable(x).to(self.device)
            self.n_query = x.size(1) - self.n_support
            # acc = self.accuracy_fun(x, n_way, i, epoch)
            acc = self.accuracy_fun(x, n_way)
            acc_all.append(acc)

            if i % 10 == 0:
                print("avg acc of task {}: {}".format(i, np.mean(acc_all)))

        acc_all = np.asarray(acc_all)
        teAcc = np.mean(acc_all)  # n_episodes 的均值
        acc_std = np.std(acc_all)
        conf_interval = 1.96 * acc_std / np.sqrt(len(test_loader))

        if epoch != -1:
            self.writer.add_scalar("testing" + '/task_acc', teAcc, epoch + 1)

        return teAcc, conf_interval
