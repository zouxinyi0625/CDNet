import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from methods.resnet import BasicBlock, ResNet

from tensorboardX import SummaryWriter
from methods.utils import write_results
from methods.discriminator import Discriminator


class L2DNet(nn.Module):
    def __init__(self, args):
        super(L2DNet, self).__init__()
        self.device = "cuda"
        self.lr = args.lr
        self.args = args
        self.method = "L2DNet_pre"
        self.n_support = args.n_shot
        self.n_query = args.n_query

        self.out_dim = args.out_dim
        self.lr = args.lr

        # encoder
        num_blocks = [2, 2, 2, 2]
        num_classes = args.n_base_class
        self.net = ResNet(BasicBlock, num_blocks, num_classes)
        self.net = self.net.cuda()

        # self.over_fineTune = args.over_fineTune

        # decomposition
        self.colors = args.color
        self.feat_dim = self.out_dim  # primary dim

        self.decomposition = nn.Sequential(
            nn.Linear(self.out_dim, self.feat_dim),
            nn.BatchNorm1d(self.feat_dim),
            nn.PReLU()
        )

        # weightnet
        if self.args.weight == "net":
            from methods.weightnet import WeightNet
            self.weight_net = WeightNet(self.out_dim + self.feat_dim)
        elif self.args.weight == "cos":
            from methods.simweight import WeightNet
            self.weight_net = WeightNet()

        # domain classifier
        self.discriminator = Discriminator(self.feat_dim, args.n_domains)

        self.n_way = args.train_n_way

        self.loss_fn = nn.CrossEntropyLoss()

        self.tf_writer = SummaryWriter(args.log_dir)

        if args.fix_d:
            self.optimizer = Adam([{'params': self.net.parameters()},
                                   {'params': self.weight_net.parameters()},
                                   {'params': self.discriminator.parameters()}],
                                  lr=1e-4)
        else:
            self.optimizer = Adam([{'params': self.net.parameters()},
                                   {'params': self.decomposition.parameters()},
                                   {'params': self.weight_net.parameters()},
                                   {'params': self.discriminator.parameters()}],
                                  lr=1e-4)

    def train_loop(self, epoch, train_loader, model_t):
        print_freq = len(train_loader) // 10
        avg_loss_ce = 0
        avg_loss_domain = 0
        avg_loss_t = 0
        for i, (x, _) in enumerate(train_loader):
            cur_domain = train_loader.batch_sampler.cur_domain
            _, loss_ce, loss_domain, loss_t = self.set_forward_loss(x, cur_domain, model_t)

            loss = loss_ce + self.args.w_d * loss_domain + self.args.w_t * loss_t

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            avg_loss_ce = avg_loss_ce + loss_ce.item()
            avg_loss_domain = avg_loss_domain + loss_domain.item()
            avg_loss_t = avg_loss_t + loss_t.item()

            if (i + 1) % print_freq == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss_ce {:f}'.format(epoch, i + 1, len(train_loader),
                                                                           avg_loss_ce / float(i + 1)))
                print('Epoch {:d} | Batch {:d}/{:d} | Loss_domain {:f}'.format(epoch, i + 1, len(train_loader),
                                                                               avg_loss_domain / float(i + 1)))
                print('Epoch {:d} | Batch {:d}/{:d} | Loss_t {:f}'.format(epoch, i + 1, len(train_loader),
                                                                          avg_loss_t / float(i + 1)))

            # if (total_it + 1) % 10 == 0 and self.tf_writer is not None:
            #     self.tf_writer.add_scalar('train/query_loss', avg_loss / float(i + 1), total_it + 1)
            #
            # total_it += 1

        print("(Avg) Epoch {:d}, Loss_ce {:f}".format(epoch, avg_loss_ce / len(train_loader)))
        print("(Avg) Epoch {:d}, Loss_domain {:f}".format(epoch, avg_loss_domain / len(train_loader)))
        print("(Avg) Epoch {:d}, Loss_t {:f}".format(epoch, avg_loss_t / len(train_loader)))

        self.tf_writer.add_scalar(self.method + '/query_loss', avg_loss_ce / len(train_loader), epoch)
        self.tf_writer.add_scalar(self.method + '/domain_loss', avg_loss_domain / len(train_loader), epoch)
        self.tf_writer.add_scalar(self.method + '/t_loss', avg_loss_t / len(train_loader), epoch)

    def extract_feature(self, x):
        x = self.net(x)  # ori feature
        ori_x = x

        re_x = 0.0

        primarys = []
        # primarys_t = []
        weights = []

        for i in range(self.colors):
            # decompostion
            primary = self.decomposition(x)

            if self.args.weight == "net":
                input_f = torch.cat([x, primary], dim=1)
                weight = self.weight_net(input_f)
            elif self.args.weight == "cos":
                weight = self.weight_net(x, primary)
                weight = weight.unsqueeze(1)

            primarys.append(primary)
            weights.append(weight)

            x = x - primary * weight

            # reconstruct
            re_x += primary * weight

        primarys = torch.stack(primarys)  # [color, dim]
        weights = torch.stack(weights)  # [color,batch,1]

        return re_x, x, ori_x, primarys, weights

    def parse_feature(self, x, model):
        x = x.cuda()

        x = x.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:])

        z_all, decomposed, z_all_ori, primarys, _ = model.extract_feature(x)

        z_all = z_all.view(self.n_way, self.n_support + self.n_query, -1)

        z_support = z_all[:, :self.n_support]
        z_query = z_all[:, self.n_support:]

        if self.args.all:
            # 若是全部监督 则primary 用z_all来表示
            primarys = z_all

        return z_support, z_query, decomposed, primarys

    def set_forward(self, x, model_t):
        z_support, z_query, decomposed, primarys = self.parse_feature(x, self)
        _, _, _, primarys_t = self.parse_feature(x, model_t)

        z_support = z_support.contiguous()
        z_proto = z_support.view(self.n_way, self.n_support, -1).mean(1)  # the shape of z is [n_data, n_dim]
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        dists = euclidean_dist(z_query, z_proto)
        scores = -dists

        scores_domain = self.discriminator(decomposed)

        return scores, scores_domain, primarys, primarys_t

    def set_forward_loss(self, x, cur_domain, model_t):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = y_query.cuda()

        scores, scores_domain, primarys, primarys_t = self.set_forward(x, model_t)

        loss = self.loss_fn(scores, y_query)

        if cur_domain != -1:  # train
            y_domain = torch.tensor([cur_domain] * self.n_way * (self.n_support + self.n_query)).cuda()
            loss_domain = self.loss_fn(scores_domain, y_domain)
        else:
            # test
            loss_domain = -1

        #  decomposition constraint
        criterion = nn.MSELoss(reduction='mean')
        loss_t = criterion(primarys_t, primarys)

        return scores, loss, loss_domain, loss_t

    def correct(self, x, cur_domain):
        scores, loss, _, _ = self.set_forward_loss(x, cur_domain, self)
        y_query = np.repeat(range(self.n_way), self.n_query)

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        return float(top1_correct), len(y_query), loss.item() * len(y_query)

    def test_loop(self, test_loader, epoch, record=None):
        loss = 0.
        count = 0

        acc_all = []

        iter_num = len(test_loader)

        for i, (x, _) in enumerate(test_loader):
            # cur_domain = test_loader.batch_sampler.cur_domain
            correct_this, count_this, loss_this = self.correct(x, cur_domain=-1)
            acc_all.append(correct_this / count_this * 100)
            loss += loss_this
            count += count_this

            acc_curent = correct_this / count_this * 100

            if (i + 1) % 10 == 0:
                print("Test Time:{}, Episode:{}, Acc:{}".format(epoch, i + 1, acc_curent))

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)

        if epoch != -1:
            self.tf_writer.add_scalar(self.method + '/test_acc', acc_mean, epoch)
            print("(Avg) Epoch {:d}, Test acc {:f}".format(epoch, acc_mean))

        print('--- %d Loss = %.6f ---' % (iter_num, loss / count))
        write_results(self.args.txt, '--- Test Time:%d Test Acc = %4.2f%% +- %4.2f%% ---' % (
            epoch, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
        #
        # print('--- Test Time:%d Test Acc = %4.2f%% +- %4.2f%% ---' % (
        #     epoch, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

        return acc_mean


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
