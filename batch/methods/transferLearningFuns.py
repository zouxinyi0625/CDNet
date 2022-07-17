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

        self.n_epoch = args.n_epoch
        self.n_class = n_class
        self.n_support = args.n_shot
        self.n_query = args.n_query

        self.out_dim = args.out_dim
        self.lr = args.lr
        self.net = net.to(self.device)
        self.over_fineTune = args.over_fineTune

        self.ft_n_epoch = args.ft_n_epoch
        self.n_way = args.n_way

        self.writer = writer

        self.base_clf = clf_fun(self, self.n_class, self.device)
        self.optimizer = Adam([{'params': self.net.parameters()},
                               {'params': self.base_clf.parameters()}],
                              lr=1e-4)
        # self.optimizer = torch.optim.Adam([{'params': self.net.parameters()},
        #                                    {'params': self.base_clf.parameters()}],
        #                                   lr=1e-4, betas=(0.5, 0.999), weight_decay=5e-4)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10, 18, 25, 32], gamma=0.1)

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

    def accuracy_fun(self, x, n_way, i_task, train_epoch):
        novel_clf = clf_fun(self, self.n_way, self.device, s=5)
        novel_optimizer = torch.optim.Adam(novel_clf.parameters(), lr=self.lr)
        x_support = x[:, :self.n_support, :, :, :].contiguous()
        x_support = x_support.view(n_way * self.n_support, *x.size()[2:])
        y_support = torch.from_numpy(np.repeat(range(n_way), self.n_support))
        y_support = Variable(y_support.to(self.device))

        x_query = x[:, self.n_support:, :, :, :].contiguous()
        x_query = x_query.view(n_way * self.n_query, *x.size()[2:])
        y_query = torch.from_numpy(np.repeat(range(n_way), self.n_query))
        y_query = Variable(y_query.cuda())

        with torch.no_grad():
            z_support = self.net(x_support)
            z_query = self.net(x_query)

        for ft_epoch in range(self.ft_n_epoch):
            loss = novel_clf.loss(novel_clf(z_support), y_support)

            novel_optimizer.zero_grad()
            loss.backward()
            novel_optimizer.step()

            # train_acc = self.acc_feature(novel_clf(z_support), y_support)
            #
            # test_acc = self.acc_feature(novel_clf(z_query), y_query)
            #
            # print("ft epoch:{}, iter:{},train_acc:{}, test_acc{}".format(ft_epoch,
            #                                                              ft_epoch + self.ft_n_epoch * i_task + 400 * 50 * train_epoch,
            #                                                              train_acc, test_acc))
            #
            # if train_epoch != -1:  # meta test
            #     self.writer.add_scalar("testing" + '/train_loss', loss.item(),
            #                            ft_epoch + self.ft_n_epoch * i_task + 400 * 50 * train_epoch)
            #     self.writer.add_scalar("testing" + '/train_acc', train_acc,
            #                            ft_epoch + self.ft_n_epoch * i_task + 400 * 50 * train_epoch)
            #     self.writer.add_scalar("testing" + '/test_acc', test_acc,
            #                            ft_epoch + self.ft_n_epoch * i_task + 400 * 50 * train_epoch)

        logits = novel_clf(z_query)
        y_hat = np.argmax(logits.data.cpu().numpy(), axis=1)
        #
        # if train_epoch != -1:
        #     self.writer.add_scalar("testing" + '/task_acc',
        #                            np.mean((y_hat == y_query.data.cpu().numpy()).astype(int)) * 100,
        #                            i_task + 1)
        #     print("Epoch:{}, task: {}, acc: {}".format(train_epoch, i_task + 1,
        #                                                np.mean(
        #                                                    (y_hat == y_query.data.cpu().numpy()).astype(int)) * 100))

        return np.mean((y_hat == y_query.data.cpu().numpy()).astype(int)) * 100

    # note: this is typical/batch based training
    def train_loop(self, trainLoader, epoch):
        self.net.train()
        loss_sum = 0
        for i, (x, y) in enumerate(trainLoader):
            x, y = Variable(x).to(self.device), Variable(y).to(self.device)
            loss = self.base_clf.loss(self.base_clf(self.net(x)), y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_sum += loss.item()

            self.writer.add_scalar("training" + '/ce_loss', loss.item(), epoch * len(trainLoader) + i + 1)

            print("Batch: {}, Iter{}, loss: {}".format(i, epoch * len(trainLoader) + i + 1, loss.item()))
        return loss_sum / len(trainLoader)

        # note: this is typical/batch based training

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
        for i, (x, _) in enumerate(test_loader):
            x = Variable(x).to(self.device)
            self.n_query = x.size(1) - self.n_support
            acc = self.accuracy_fun(x, n_way, i, epoch)
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
