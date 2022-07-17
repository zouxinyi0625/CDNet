import argparse


def args_parser(fs_approach):
    args = argparse.ArgumentParser(description='experiment mode %s' % (fs_approach))

    args.add_argument('--dataset', default='RAF')
    args.add_argument('--testset', default='CFEE')
    args.add_argument('--pretrain', action='store_true')
    args.add_argument('--backbone', default='ResNet18', help='Conv4|ResNet12|ResNet18|WideResNet')

    args.add_argument('--seed', default=10, help='the seed used for training')
    args.add_argument('--num_workers', default=12, help='the number of workers')

    args.add_argument('--data_aug', action='store_true', help='perform data augmentation or not during training')
    args.add_argument('--img_size', default=224,
                      help='input size of the backbone: 84, 224, 80  for miniImagenet, ResNet18, WideResNet')

    args.add_argument('--n_epoch', default=100, type=int, help='the last epoch for stop')
    args.add_argument('--n_iters', default=10000, type=int, help='the last epoch for stop')
    args.add_argument('--test_epoch', default=-1, type=int, help='-1: best model, or the epoch to test')
    args.add_argument('--lr', default=0.001, help='0.001 the learning rate of classifier')
    args.add_argument('--n_domains', default=5, help='number of multiple domains')
    args.add_argument('--n_batches', default=100, help='number of batches (to generate the domain list)')

    args.add_argument('--benchmarks_dir',
                      default='root to the data path/',
                      help='the directory which the benchmarks are strored')
   
    args.add_argument('--checkpoint_dir', default='root to save the model',
                      help='the directory for saving the best_model')
    args.add_argument('--results_dir', default='root to save the result',
                      help='the directory for saving the best_model')

    args.add_argument('--test_n_way', default=5, type=int, help='number of classes in each meta-validation')
    args.add_argument('--n_shot', default=5, type=int, help='support size in the episodic training literature')
    args.add_argument('--n_query', default=16, type=int,
                      help='pretented unlabled data for loss calculation in meta-learning')

    args.add_argument('--testing_epochs', default=5, help='the number of epoch for measuring the accuracy')
    args.add_argument('--n_episodes', default=600, help='the number of episodes for measuring the accuracy')

    args.add_argument('--n_way', default=5, type=int, help='number of classes in each meta-training')

    args.add_argument('--split', default="all", help="set test set")

    args.add_argument('--name', default="cascade", help="choose a model")
    args.add_argument('--k', default=256, type=int, help='principle components')
    args.add_argument('--group', default=7, type=int, help='feature groups (for decomposition)')
    args.add_argument('--color', default=3, type=int, help='primary colors (for cascade)')

    args.add_argument('--projection_dim', default=128, help='out dim for projection head in SimCLR')
    args.add_argument('--temp', type=float, default=1, help='Temperature of SIMCLR')

    args.add_argument('--teacher_name', default='root to pretrained encoder')

    args.add_argument('--w_ce', default=1.0, type=float, help='loss weight of classification (reconstructed feature)')
    args.add_argument('--w_de', default=1.0, type=float,
                      help='loss weight of expression entropy (decomposed feature)')
    args.add_argument('--w_re', default=1.0, type=float,
                      help='loss weight of domain entropy (reconstruction feature)')
    args.add_argument('--w_mi', default=1.0, type=float,
                      help='loss weight of mutual information')
    args.add_argument('--w_t', default=1.0, type=float,
                      help='loss weight of reconstruction loss (difference between reconstructed and ori feature)')
    args.add_argument('--w_domain', default=1.0, type=float,
                      help='loss weight of domain classification (decomposed feature)')
    args.add_argument('--w_c', default=0.0001, type=float,
                      help='loss weight of domain classification (decomposed feature)')
    args.add_argument('--w_pd', default=1.0, type=float,
                      help='loss weight of primary (can not distinguish domain)')

    args.add_argument('--w_base', default=1.0, type=float,
                      help='loss weight of basic expression classification in Step 2')
    args.add_argument('--w_SIMCLR', default=1.0, type=float,
                      help='loss weight of contrastive loss in Step 2')

    if fs_approach == 'meta-learning':
        args.add_argument('--method', default='ProtoNet', help='MatchingNet|ProtoNet|RelationNet{_softmax}')
        args.add_argument('--train_n_way', default=5, type=int, help='number of classes in each meta-training')
        args.add_argument('--n_support', default=5, type=int, help='number of classes in each meta-training')

    elif fs_approach == 'transfer-learning':
        args.add_argument('--over_fineTune', type=bool, default=False, help='perform over fine-tunining')
        args.add_argument('--n_base_class', default=7, help='number of base categories')
        args.add_argument('--method', default='softMax', help='softMax|cosMax|arcMax')
        args.add_argument('--batch_size', default=64, type=int, help='64 the batch size during base-training')
        args.add_argument('--ft_n_epoch', default=50, type=int,
                          help='the number of testing epochs during the novel and validation fine-tunining')
        args.add_argument('--mode', default='train', help='softMax|cosMax|arcMax')

    else:
        raise ValueError('unknown few-shot approach!')

    return args.parse_args()
