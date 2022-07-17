import torchvision.models as models
from backbones.shallow_backbone import Conv4Net, Conv4Net_RN, Flatten
from backbones.resnet import BasicBlock, ResNet
from torch import nn
import shutil
import torch
import os
import socket


def clear_temp(data_path):
    if os.path.isdir(data_path + 'temp'):
        for _, name_temp, _ in os.walk(data_path + 'temp'): break
        if name_temp != []:
            shutil.move(data_path + 'temp/' + name_temp[0], data_path)
        os.rmdir(data_path + '/temp')


def device_kwargs(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    return device


def backboneSet(args, fs_approach):
    if args.dataset == 'miniImagenet':
        args.n_base_class = 64
    elif args.dataset == 'CUB':
        args.n_base_class = 100
    elif args.dataset == 'RAF':
        args.n_base_class = 7
    elif args.dataset == 'CFEE_B':
        args.n_base_class = 7
    elif args.dataset == 'emotion_B':
        args.n_base_class = 7
    elif args.dataset == 'multi':
        args.n_base_class = 8
    else:
        raise "ar: sepcify the number of base categories!"

    if args.backbone == "Conv4":
        if args.method in ['RelationNet', 'RelationNet_softmax']:
            args.out_dim = [64, 19, 19]
            net = Conv4Net_RN()
            args.img_size = 84
        else:
            net = Conv4Net()
            args.out_dim = 1600
            args.img_size = 84

    elif args.backbone == "ResNet18":
        num_blocks = [2, 2, 2, 2]
        num_classes = args.n_base_class
        net = ResNet(BasicBlock, num_blocks, num_classes)
        net = net.cuda()

        args.img_size = 224
        args.out_dim = 512

        if args.pretrain:
            # 加载初始化参数
            # face_pretrained MS_CelebA
            hostname = socket.gethostname()
            if hostname == "yanyan3":
                model_path = "/data/zxy_data/model/face_pretrained.pth.tar"
            if hostname == "yanyan1":
                model_path = "/media/data/zxy_data/model/face_pretrained.pth.tar"
            if hostname == "zxy-MS":
                model_path = "/home/zxy/data/model/face_pretrained.pth.tar"

            pretrained = torch.load(model_path)
            pretrained_state_dict = pretrained['state_dict']
            model_state_dict = net.state_dict()  # 当前model的key

            # 去掉fc层参数
            for key in pretrained_state_dict:
                if key in ['module.fc.weight', 'module.fc.bias', 'module.feature.weight', 'module.feature.bias']:
                    pass
                else:
                    model_state_dict[key.replace("module.", "")] = pretrained_state_dict[key]

            net.load_state_dict(model_state_dict)

    # set file name
    file_name = str(args.test_n_way) + 'way_' + str(
        args.n_shot) + 'shot_' + args.dataset + '_' + args.testset + '_' + args.split + '_' + args.name

    # cascade 分解(表情分类 域信息 teacher监督) finetune (有标签新类上微调w, cls) cascade_c (+compact loss) unlabel (加入无标签图像上的子监督损失)
    # decomposition 完整的包含所有损失
    if args.name == "cascade" or args.name == "finetune" or args.name == "cascade_c" or args.name == "decomposition" or args.name == "cascade_t" or args.name == "cascade_e" or args.name == "cascade_pa":
        # file_name += '_' + str(args.color) + "_" + str(args.w_ce) + "_" + str(args.w_de) + "_" + str(
        #     args.w_t) + "_" + str(
        #     args.w_domain)
        file_name += '_' + str(args.color) + "_" + str(args.w_ce) + "_" + str(args.w_domain)

        if args.name == "cascade_c":
            file_name += "_" + str(args.w_c)

        if args.name == "decomposition":
            file_name += "_" + str(args.w_c) + "_" + str(args.w_pd)

    if args.name == "cascade_mi":
        file_name += '_' + str(args.color) + "_" + str(args.w_ce) + "_" + str(args.w_mi)
    #
    # elif args.name == "unlabel":
    #     file_name += '_' + str(args.color) + "_" + str(args.w_ce) + "_" + str(args.w_de) + "_" + str(
    #         args.w_t) + "_" + str(
    #         args.w_domain)

    file_name += "_" + args.method + '_' + args.backbone

    return args, net, file_name
