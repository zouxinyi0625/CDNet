from data.utils import ImageJitter
from torchvision.transforms import Compose
import torchvision.transforms as Transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
from torch.utils.data import Dataset
import os


class ar_dataset_open(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.file_list = os.listdir(root)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.file_list[idx])
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = self.transform(img)
        return img


def ar_transform(args, aug):
    norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    jitter_param = dict(Brightness=0.4, Contrast=0.4, Color=0.4, )

    if aug:
        transforms = Compose([Transforms.RandomResizedCrop(args.img_size),
                              ImageJitter(jitter_param),
                              Transforms.RandomHorizontalFlip(),
                              Transforms.ToTensor(),
                              Transforms.Normalize(norm_mean, norm_std)])
        # transforms = Compose([
        #     Transforms.Resize(256),
        #     Transforms.RandomCrop(224),
        #     Transforms.RandomHorizontalFlip(),
        #     Transforms.ToTensor(),
        #     Transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        # ])
    else:
        # transforms = Compose([Transforms.RandomResizedCrop(args.img_size),
        #                       ImageJitter(jitter_param),
        #                       Transforms.ToTensor(),
        #                       Transforms.Normalize(norm_mean, norm_std)])
        transforms = Compose([
            Transforms.Resize(256),
            Transforms.RandomCrop(224),
            Transforms.ToTensor(),
            Transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    return transforms


def unlabel_DataLoader(args, aug, section='base', shuffle=True):
    data_path = args.benchmarks_dir + args.testset + '/images'
    if args.split == "val":
        data_path += "/test/"
    elif args.split == "all":
        data_path += "/all/"
    print("loading unlabeled data from {}".format(data_path))

    transforms = ar_transform(args, aug)
    dataset = ImageFolder(root=data_path, transform=transforms)
    return DataLoader(dataset=dataset,
                      batch_size=args.batch_size,
                      num_workers=args.num_workers,
                      shuffle=shuffle,
                      drop_last=False)


def ar_base_DataLaoder(args, aug, section='base', shuffle=True):
    # data_path = args.benchmarks_dir + args.dataset + '/images/' + section + '/'
    data_path = args.benchmarks_dir + args.dataset + "/images/all/"
    print("({} batch) loading data from {}".format(section, data_path))

    transforms = ar_transform(args, aug)
    dataset = ImageFolder(root=data_path, transform=transforms)
    return DataLoader(dataset=dataset,
                      batch_size=args.batch_size,
                      num_workers=args.num_workers,
                      shuffle=shuffle,
                      drop_last=False)


def visualize_DataLaoder(args, dataset, aug, section='base', shuffle=True):
    # data_path = args.benchmarks_dir + args.dataset + '/images/' + section + '/'
    data_path = args.benchmarks_dir + dataset + '/images/' + section + '/'
    print("(Visualize) loading data from {}".format(data_path))

    transforms = ar_transform(args, aug)
    dataset = ImageFolder(root=data_path, transform=transforms)
    return DataLoader(dataset=dataset,
                      batch_size=args.batch_size,
                      num_workers=args.num_workers,
                      shuffle=shuffle,
                      drop_last=False)


def ar_base_underFolder_DataLaoder(args, aug, section='base_undreFolder'):
    data_path = args.benchmarks_dir + args.dataset + '/' + section + '/'
    transforms = ar_transform(args, aug)

    loaderList = []
    for i in range(args.n_base_class):
        dataset = ImageFolder(root=data_path, transform=transforms)
        loaderList.append(DataLoader(dataset=dataset,
                                     batch_size=args.n_shot,
                                     num_workers=args.num_workers,
                                     shuffle=True,
                                     drop_last=False))

    return loaderList
