import numpy as np
import torch
import torch.optim
import os

from methods import backbone
from methods.backbone import model_dict
from data.datamgr import SimpleDataManager, SetDataManager

from options import parse_args, get_resume_file, load_warmup_state
from methods.utils import Setargs, write_results
import socket
from methods.L2DNet_pre import L2DNet


def train(base_loader, val_loader, model, start_epoch, stop_epoch, params, model_t):
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    # for validation
    max_acc = 0
    best_epoch = -1
    
    for epoch in range(start_epoch, stop_epoch):
        model.train()
        model.train_loop(epoch, base_loader, model_t)

        model.eval()
        acc = model.test_loop(val_loader, epoch)

        if acc > max_acc:
            best_epoch = epoch
            write_results(params.txt, "best model! save... at epoch {}".format(best_epoch))
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile, _use_new_zipfile_serialization=False)
        else:
            write_results(params.txt, "GG! best accuracy {:f} at {}".format(max_acc, best_epoch))

        if ((epoch + 1) % params.save_freq == 0) or (epoch == stop_epoch - 1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile, _use_new_zipfile_serialization=False)

    return model


def test(model, params):
    checkpoint_path = os.path.join(params.checkpoint_dir, "best_model.tar")
    # if params.n_shot == 1:
    #     checkpoint_path = checkpoint_path.replace('1shot', '5shot')

    print("load saved model from {}".format(checkpoint_path))

    checkpoint = torch.load(os.path.join(checkpoint_path))

    model.load_state_dict(checkpoint['state'])

    best_epoch = checkpoint['epoch'] + 1
    write_results(params.txt, 'testing... best model at epoch %d' % (best_epoch))

    val_file = os.path.join(params.data_dir, params.testset, params.split + '.json')
    novel_datamgr = SetDataManager(image_size, n_query=params.n_query, **test_few_shot_params, n_eposide=1000)
    noLoader = novel_datamgr.get_data_loader(val_file, aug=False)

    model.eval()
    # acc, confInt = [], []
    for epoch in range(5):
        model.test_loop(noLoader, epoch=-1)


# --- main function ---
if __name__ == '__main__':

    # set numpy random seed
    np.random.seed(10)

    # parser argument
    params = parse_args('train')

    # change data path & save root
    teacher_dir = params.teacher_dir

    # set other params
    params = Setargs(params)

    params.txt = os.path.join(params.save_dir, "txt", params.file_name + '.txt')

    print(params)

    print('\n--- l2d training: {} ---\n'.format(params.file_name))

    # output and tensorboard dir
    params.log_dir = '%s/log/%s' % (params.save_dir, params.file_name)
    params.checkpoint_dir = '%s/checkpoints/%s' % (params.save_dir, params.file_name)
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    # dataloader
    print('\n--- prepare dataloader ---')
    if params.dataset == 'multi':
        print('  train with multiple seen domains (unseen domain: {})'.format(params.testset))
        datasets = ['RAF', 'CK+', 'MMI', 'OULU', 'SFEW']

        base_file = [os.path.join(params.data_dir, dataset, 'all.json') for dataset in datasets]
        val_file = os.path.join(params.data_dir, params.testset, params.split + '.json')
    else:
        print('  train with single seen domain {}'.format(params.dataset))
        base_file = os.path.join(params.data_dir, params.dataset, 'all.json')
        val_file = os.path.join(params.data_dir, params.testset, 'all.json')

    print("base file:", base_file)
    print("val file:", val_file)

    print('\n--- build dataset ---')

    image_size = 224

    train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot)
    base_datamgr = SetDataManager(image_size, n_query=params.n_query, **train_few_shot_params)
    base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)

    test_few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
    val_datamgr = SetDataManager(image_size, n_query=params.n_query, **test_few_shot_params, n_eposide=400)
    val_loader = val_datamgr.get_data_loader(val_file, aug=False)

    print('\n--- build model ---')

    # student
    model = L2DNet(params)
    model = model.cuda()

    # teacher
    model_t = L2DNet(params)
    model_t = model_t.cuda()

    # load teacher model
    pretrain_name = "5way_" + str(
        params.n_shot) + "shot_multi_" + params.testset + "_" + params.split + "_cascade_e_" + str(
        params.color) + "_1.0_1.0_softMax_ResNet18"

    pretrain_path = os.path.join(teacher_dir, pretrain_name, "best_model.tar")

    print("loading pretrained model from {}".format(pretrain_path))

    temp = torch.load(pretrain_path)

    model_t.net.load_state_dict(temp["state"])
    model_t.decomposition.load_state_dict(temp['decomposition'])
    model_t.weight_net.load_state_dict(temp['weight_net'])
    model_t.discriminator.load_state_dict(temp['discriminator'], strict=False)

    if params.pretrain:
        model.net.load_state_dict(temp["state"])
        model.decomposition.load_state_dict(temp['decomposition'])
        model.weight_net.load_state_dict(temp['weight_net'])
        model.discriminator.load_state_dict(temp['discriminator'], strict=False)
    else:
        # face_pretrained MS_CelebA
        print("without pretrain!")
        hostname = socket.gethostname()
        model_path = params.pretrain_model
        pretrained = torch.load(model_path)
        pretrained_state_dict = pretrained['state_dict']
        model_state_dict = model.net.state_dict()  # 当前model的key

        # 去掉fc层参数
        for key in pretrained_state_dict:
            if key in ['module.fc.weight', 'module.fc.bias', 'module.feature.weight', 'module.feature.bias']:
                pass
            else:
                model_state_dict[key.replace("module.", "")] = pretrained_state_dict[key]

        model.net.load_state_dict(model_state_dict)

    # set epoch
    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch

    # training
    print('\n--- start the training ---')
    model = train(base_loader, val_loader, model, start_epoch, stop_epoch, params, model_t)

    # test
    write_results(params.txt, "---- Load best model and test ----")
    test(model, params)

    write_results(params.txt, "---- The END ----")
