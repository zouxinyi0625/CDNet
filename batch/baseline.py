from __future__ import print_function
import os
import torch
import numpy as np
from args_parser import args_parser
from backbones.utils import clear_temp
from backbones.utils import backboneSet
from data.tl_dataFunctions import ar_base_DataLaoder, unlabel_DataLoader
from data.ml_dataFunctions import SetDataManager
from data.multi_dataFunctions import create_dataloaders
from utils import write_results
import socket
from utils import get_learning_rate
from methods.transferLearningFuns import transferLearningFuns
from tensorboardX import SummaryWriter
from datetime import datetime
import random


def meta_training(args, model, file_name, resume, file):
    clear_temp(args.benchmarks_dir + args.dataset + '/base/')
    print('transfer training...')
    checkpoint_dir = os.path.join(args.checkpoint_dir, file_name)

    max_acc = 0
    first_epoch = 0

    # make dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if resume and os.path.isfile(checkpoint_dir):  # 加载先前训练的模型
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, file_name, "best_model.tar"))
        first_epoch = checkpoint['epoch'] + 1
        model.net.load_state_dict(checkpoint['state'])
        max_acc = checkpoint['max_acc']
        model.optimizer = checkpoint['optimizer']
        model.base_clf = checkpoint['base_clf']
        model.base_clf.train()
        model.net.train()
        print('resume: up to now the best model has:', str(max_acc), 'accuracy at ep.', first_epoch, '!')

    # labeled base FER data
    dataset = ['CK+', "OULU", "MMI", "RAF", "SFEW"]
    trLoaders = create_dataloaders(args, dataset)  # 每个数据集构造一个dataloader

    val_file = args.benchmarks_dir + args.testset + "/" + args.split + '.json'
    write_results(file, "(val) loading data from {}".format(val_file))

    val_datamgr = SetDataManager(args.img_size, args.test_n_way, args.n_shot, args.n_query, n_episodes=400)
    vaLoader = val_datamgr.get_data_loader(val_file, aug=False)

    # random domain list
    domain_list = [i % args.n_domains for i in range(args.n_epoch)]
    random.shuffle(domain_list)

    for epoch in range(first_epoch, args.n_epoch):
        model.net.train()
        lr = get_learning_rate(model.optimizer)

        rand_domain = domain_list[epoch]
        trLoader = trLoaders[rand_domain]

        loss = model.train_loop(trLoader, epoch)
        write_results(file_txt,
                      "Epoch:{}, lr_net:{}, lr_clf:{}, loss:{}, domain:{}".format(epoch, lr[0], lr[1], loss,
                                                                                  dataset[rand_domain]))

        model.net.eval()

        vaAcc, vaConf_interval = model.test_loop(vaLoader, args.test_n_way, epoch)

        print_txt = 'epoch %d Loss: %4.2f%% ||| vaAcc: %4.2f%% +- %4.2f%% |||'

        if (epoch + 1) % 10 == 0:
            torch.save({'epoch': epoch,
                        'state': model.net.state_dict(),
                        'max_acc': vaAcc,
                        'base_clf': model.base_clf,
                        'optimizer': model.optimizer},
                       os.path.join(checkpoint_dir, "{}.tar".format(str(epoch))))

        if vaAcc > max_acc:
            max_acc = vaAcc

            torch.save({'epoch': epoch,
                        'state': model.net.state_dict(),
                        'max_acc': max_acc,
                        'base_clf': model.base_clf,
                        'optimizer': model.optimizer},
                       os.path.join(checkpoint_dir, "best_model.tar"))

            print_txt = print_txt + ' update...'

        write_results(file, print_txt % (epoch, loss, vaAcc, vaConf_interval))

        # model.scheduler.step()


def meta_testing(args, model, file_name, file, partition='novel'):
    checkpoint = torch.load(os.path.join(args.checkpoint_dir, file_name, "best_model.tar"))
    model.net.load_state_dict(checkpoint['state'])

    best_epoch = checkpoint['epoch'] + 1
    write_results(file, 'transfer testing... best model at epoch %d' % (best_epoch))

    novel_file = args.benchmarks_dir + args.testset + "/" + args.split + '.json'
    write_results(file, "(novel) loading data from {}".format(novel_file))

    novel_datamgr = SetDataManager(args.img_size, args.test_n_way, args.n_shot, args.n_query, n_episodes=1000)
    noLoader = novel_datamgr.get_data_loader(novel_file, aug=False)
    model.net.eval()
    acc, confInt = [], []
    for epoch in range(args.testing_epochs):
        noAcc, noConf_interval = model.test_loop(noLoader, args.test_n_way, -1)
        acc.append(noAcc)
        confInt.append(noConf_interval)
        write_results(file, 'meta-testing acc: %4.2f%% +- %4.2f%%' % (np.average(acc), np.average(confInt)))


if __name__ == '__main__':
    fs_approach = 'transfer-learning'
    args = args_parser(fs_approach)
    args, net, file_dir = backboneSet(args, fs_approach)
    file_txt = file_dir + ".txt"
    file_txt = os.path.join(args.results_dir, file_txt)

    # TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    args.log_dir = args.log_dir + file_dir
    writer = SummaryWriter(args.log_dir)

    print(args)

    model = transferLearningFuns(args, net, args.n_base_class, writer)  # 初始化模型
    model = model.cuda()
    meta_training(args, model, file_dir, resume=False, file=file_txt)  # 预训练
    meta_testing(args, model, file_dir, file=file_txt)  # 微调
