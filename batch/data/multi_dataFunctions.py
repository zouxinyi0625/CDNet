from data.tl_dataFunctions import ar_transform
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import torch
import os, random

label2class = {
    0: "neutral",
    1: "anger",
    2: "surprise",
    3: "disgust",
    4: "afraid",
    5: "happy",
    6: "sadness",
    7: "contempt"
}


class FERData(Dataset):
    # name 数据集的名字 root 数据存放的未知 mode train/test
    def __init__(self, data_path, name, args, aug=True):
        self.transform = ar_transform(args, aug)
        self.images = []
        for root, dirs, files in os.walk(data_path):  # 遍历统计
            for file in files:
                self.images.append(os.path.join(root, file))
        self.name = name

    def __getitem__(self, index):
        img_path = self.images[index]

        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        ori_label = img_path.split(os.sep)[-2]
        label = self.change_label(int(ori_label))
        # print(ori_label, label, label2class[label])
        return img, torch.tensor(label)

    def change_label(self, ori):
        if self.name == "CK+":
            map = [1, 2, 3, 4, 5, 6, 7]
        if self.name == "RAF":
            map = [2, 4, 3, 5, 6, 1, 0]
        if self.name == "OULU":
            map = [1, 2, 3, 4, 5, 6]
        if self.name == "MMI":
            map = [1, 2, 3, 4, 5, 6]
        if self.name == "SFEW":
            map = [0, 1, 2, 3, 4, 5, 6]
        return map[ori]

    def __len__(self):
        return len(self.images)


def create_dataloaders(args, dataset):
    dataloaders = []
    for name in dataset:
        data_path = args.benchmarks_dir + name + '/images/all'
        dataset = FERData(data_path, name, args)
        dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)
        dataloaders.append(dataloader)
    return dataloaders


if __name__ == "__main__":
    root = "/media/data1/zxy_data/FER_zxy/"
    dataset = ["CK+", "RAF", "SFEW", "OULU", "MMI"]
    batch_size = 4
    dataloaders = create_dataloaders(root, dataset, batch_size)

    num_classes = 8

    for epoch in range(10):
        # 生成随机domain
        rand_domain = random.randint(0, 4)
        print(rand_domain, dataset[rand_domain])
        x, y = next(iter(dataloaders[rand_domain]))
