from methods.transferLearning_clfHeads import softMax, cosMax, arcMax
from torch.optim import Adam
import torch
from torch import nn
from torch.nn import functional as F


def clf_optimizer(args, net, device, frozen_net, s=15, m=0.01):
    if frozen_net: s = 5
    if args.method == 'softMax':
        clf = softMax(args.out_dim, args.test_n_way).to(device)
    elif args.method == 'cosMax':
        clf = cosMax(args.out_dim, args.test_n_way, s).to(device)
    elif args.method == 'arcMax':
        clf = arcMax(args.out_dim, args.test_n_way, s, m).to(device)
    if frozen_net:
        optimizer = torch.optim.Adam(clf.parameters(), lr=args.lr)
    else:
        optimizer = Adam([{'params': net.parameters()},
                          {'params': clf.parameters()}],
                         lr=args.lr)
    return clf, optimizer


class projector_SIMCLR(nn.Module):
    '''
        The projector for SimCLR. This is added on top of a backbone for SimCLR Training
    '''

    def __init__(self, in_dim, out_dim):
        super(projector_SIMCLR, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fc1 = nn.Linear(in_dim, in_dim)
        self.fc2 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class apply_twice:
    '''
        A wrapper for torchvision transform. The transform is applied twice for
        SimCLR training
    '''

    def __init__(self, transform, transform2=None):
        self.transform = transform

        if transform2 is not None:
            self.transform2 = transform2
        else:
            self.transform2 = transform

    def __call__(self, img):
        return self.transform(img), self.transform2(img)


def pseudolabel_dataset(embedding, clf, dataset, transform, transform_test, params):
    '''
        pseudolabel the dataset with the teacher model (embedding, clf)
    '''

    # Change the transform of the target dataset to the deterministic transformation
    dataset.d.transform = transform_test
    dataset.d.target_transform = (lambda x: x)

    embedding.eval()
    clf.eval()

    loader = torch.utils.data.DataLoader(dataset, batch_size=params.batch_size,
                                         shuffle=False, drop_last=False, num_workers=params.num_workers)

    # do an inference on the full target dataset
    probs_all = []
    for X, _ in loader:
        X = X.cuda()

        with torch.no_grad():
            feature = embedding(X)
            logits = clf(feature)
            probs = F.softmax(logits, dim=1)
            probs += 1e-6

        probs_all.append(probs)

    probs_all = torch.cat(probs_all, dim=0).cpu()

    # Update the target dataset with the pseudolabel
    if hasattr(dataset.d, 'targets'):
        dataset.d.targets = probs_all
        samples = [(i[0], probs_all[ind_i]) for ind_i, i in enumerate(dataset.d.samples)]
        dataset.d.samples = samples
        dataset.d.imgs = samples
    elif hasattr(dataset.d, "labels"):
        dataset.d.labels = probs_all
    else:
        raise ValueError("No Targets variable found!")

    # Switch the dataset's augmentation back to the stochastic augmentation
    dataset.d.transform = transform
    return dataset
