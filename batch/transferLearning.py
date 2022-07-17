from __future__ import print_function
import os
import torch
import numpy as np
from args_parser import args_parser
from backbones.utils import clear_temp
from backbones.utils import backboneSet
from data.tl_dataFunctions import ar_base_DataLaoder
from data.ml_dataFunctions import SetDataManager
from utils import write_results
import socket
from utils import get_learning_rate


def meta_training(args, model, file_name, resume, file):
    clear_temp(args.benchmarks_dir + args.dataset + '/base/')
    print('transfer training...')
    checkpoint_dir = os.path.join(args.checkpoint_dir, file_name)

    max_acc = 0
    first_epoch = 0
    if resume and os.path.isfile(checkpoint_dir):  # 加载先前训练的模型
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, file_name))
        first_epoch = checkpoint['epoch'] + 1
        model.net.load_state_dict(checkpoint['state'])
        max_acc = checkpoint['max_acc']
        model.optimizer = checkpoint['optimizer']
        model.base_clf = checkpoint['base_clf']
        model.base_clf.train()
        model.net.train()
        print('resume: up to now the best model has:', str(max_acc), 'accuracy at ep.', first_epoch, '!')

    # conventional FER task
    trLoader = ar_base_DataLaoder(args, aug=True, shuffle=True, section="train")
    # testLoader = ar_base_DataLaoder(args, aug=False, shuffle=False, section="test")

    val_file = args.benchmarks_dir + args.testset + "/" + args.split + '.json'
    write_results(file, "(val) loading data from {}".format(val_file))

    val_datamgr = SetDataManager(args.img_size, args.test_n_way, args.n_shot, args.n_query, n_episodes=400)
    vaLoader = val_datamgr.get_data_loader(val_file, aug=False)
    for epoch in range(first_epoch, args.n_epoch):
        model.net.train()
        lr = get_learning_rate(model.optimizer)

        loss = model.train_loop(trLoader)
        # feracc = model.test_fer(testLoader)
        # write_results(file_txt,
        #               "Epoch:{}, lr_net:{}, lr_clf:{}, loss:{}, feracc:{}".format(epoch, lr[0], lr[1], loss, feracc))
        write_results(file_txt,
                      "Epoch:{}, lr_net:{}, lr_clf:{}, loss:{}".format(epoch, lr[0], lr[1], loss))

        # model.net.eval()
        if args.name == "base_FA":
            vaAcc, vaConf_interval, novel_net = model.test_loop(vaLoader, args.test_n_way)
        else:
            vaAcc, vaConf_interval = model.test_loop(vaLoader, args.test_n_way)

        print_txt = 'epoch %d Loss: %4.2f%% ||| vaAcc: %4.2f%% +- %4.2f%% |||'
        if vaAcc > max_acc:
            max_acc = vaAcc
            if args.name == "cascade":
                torch.save({'epoch': epoch,
                            'state': model.net.state_dict(),
                            'decompose': model.fc_group.state_dict(),
                            'reconstruct': model.self_attn_weight.state_dict(),
                            'max_acc': max_acc,
                            'base_clf': model.base_clf,
                            'optimizer': model.optimizer},
                           checkpoint_dir)
            if args.name == "base_FA":
                torch.save({'epoch': epoch,
                            'state': model.net.state_dict(),
                            'state_novel': novel_net.state_dict(),
                            'max_acc': max_acc,
                            'base_clf': model.base_clf,
                            'optimizer': model.optimizer},
                           checkpoint_dir)
            else:
                torch.save({'epoch': epoch,
                            'state': model.net.state_dict(),
                            'max_acc': max_acc,
                            'base_clf': model.base_clf,
                            'optimizer': model.optimizer},
                           checkpoint_dir)

            print_txt = print_txt + ' update...'

        write_results(file, print_txt % (epoch, loss, vaAcc, vaConf_interval))

        # model.scheduler.step()


def meta_testing(args, model, file_name, file, partition='novel'):
    checkpoint = torch.load(os.path.join(args.checkpoint_dir, file_name))
    model.net.load_state_dict(checkpoint['state'])
    if args.name == "cascade":
        model.fc_group.load_state_dict(checkpoint['decompose'])
    best_epoch = checkpoint['epoch'] + 1
    write_results(file, 'transfer testing... best model at epoch %d' % (best_epoch))

    novel_file = args.benchmarks_dir + args.testset + "/" + args.split + '.json'
    write_results(file, "(novel) loading data from {}".format(novel_file))

    novel_datamgr = SetDataManager(args.img_size, args.test_n_way, args.n_shot, args.n_query, n_episodes=1000)
    noLoader = novel_datamgr.get_data_loader(novel_file, aug=False)
    model.net.eval()
    acc, confInt = [], []
    for epoch in range(args.testing_epochs):
        noAcc, noConf_interval = model.test_loop(noLoader, args.test_n_way)
        acc.append(noAcc)
        confInt.append(noConf_interval)
        write_results(file, 'meta-testing acc: %4.2f%% +- %4.2f%%' % (np.average(acc), np.average(confInt)))


if __name__ == '__main__':
    fs_approach = 'transfer-learning'
    args = args_parser(fs_approach)
    
    args, net, file_name = backboneSet(args, fs_approach)
    file_txt = file_name.replace(".tar", ".txt")
    file_txt = os.path.join(args.results_dir, file_txt)

    print(args)

    from methods.transferLearningFuns import transferLearningFuns


    model = transferLearningFuns(args, net, args.n_base_class)  # 初始化模型
    model = model.cuda()
    meta_training(args, model, file_name, resume=False, file=file_txt)  # 预训练
    meta_testing(args, model, file_name, file=file_txt)  # 微调
