def Setargs(args):
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

    # set file name
    file_name = str(args.test_n_way) + 'way_' + str(
        args.n_shot) + 'shot_' + args.dataset + '_' + args.testset + '_' + args.split + '_' + args.name

    if args.name == "cascade":
        # 仅分解 (Contribution 1)
        file_name += '_' + str(args.color) + '_' + str(args.w_d)

        if args.fix_e:
            file_name += '_fixe'

    if args.name == "cascade_domain":
        file_name += '_' + str(args.color) + '_' + str(args.weight) + '_' + str(args.w_d)

    if args.name == "cascade_pre" or args.name == "cascade_pre_t" or args.name == "cascade_pre_pa":
        file_name += '_' + str(args.color) + '_' + str(args.w_d) + '_' + str(args.w_t)

    if args.name == "cascade_pre_aug":
        file_name += '_' + str(args.color) + '_' + str(args.weight) + '_' + str(args.w_d) + '_' + str(
            args.w_t) + '_' + str(args.noise_num) + "_" + str(args.eps)

    if args.name == "cascade_aug":
        file_name += '_' + str(args.color) + '_' + str(args.weight) + '_' + str(args.noise_num) + "_" + str(args.eps)
        if args.wori:
            file_name += "_" + "wori"

    # if args.name == "cascade" or args.name == "finetune" or args.name == "cascade_c":
    #     file_name += '_' + str(args.color) + "_" + str(args.w_ce) + "_" + str(args.w_de)

    if args.pretrain:
        file_name += '_pre'

    if args.fix_d:
        file_name += '_fixd'

    if args.lr_decay:
        file_name += '_lr'

    #  全部监督
    if args.all:
        file_name += '_all'

    args.file_name = file_name

    return args


def write_results(file, txt):
    with open(file, 'a') as f:
        f.write(txt)
        f.write('\n')
    print(txt)
