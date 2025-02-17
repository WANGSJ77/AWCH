import torch
import torchvision
from .random_dataset import RandomDataset
from datasets.MyDataset import MyCustomDataset

def get_dataset(dataset, data_dir, transform, train=True, download=False, debug_subset_size=None):
    if dataset == 'mnist':
        dataset = torchvision.datasets.MNIST(data_dir, train=train, transform=transform, download=download)
    elif dataset == 'stl10':
        dataset = torchvision.datasets.STL10(data_dir, split='train+unlabeled' if train else 'test', transform=transform, download=download)
    elif dataset == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(data_dir, train=train, transform=transform, download=download)
    elif dataset == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(data_dir, train=train, transform=transform, download=download)
    elif dataset == 'imagenet':
        dataset = torchvision.datasets.ImageNet(data_dir, split='train' if train == True else 'val', transform=transform, download=download)
    elif dataset == 'random':
        dataset = RandomDataset()
    elif dataset == 'UCM':
        if train==True:
            dataset = MyCustomDataset(root_path="D:/Wang_SiJia/SelfSupervised-main/Data/UCM/train.txt",
                                      class_path='D:/Wang_SiJia/SelfSupervised-main/Data/UCM/class.txt',
                                      transform=transform)
        else:
            dataset = MyCustomDataset(root_path="D:/Wang_SiJia/SelfSupervised-main/Data/UCM/test.txt",
                                      class_path='D:/Wang_SiJia/SelfSupervised-main/Data/UCM/class.txt',
                                      transform=transform)
    elif dataset == 'AID':
        if train==True:
            dataset = MyCustomDataset(root_path="D:/Wang_SiJia/SelfSupervised-main/Data/AID/train.txt",
                                      class_path='D:/Wang_SiJia/SelfSupervised-main/Data/AID/class.txt',
                                      transform=transform)
        else:
            dataset = MyCustomDataset(root_path="D:/Wang_SiJia/SelfSupervised-main/Data/AID/test.txt",
                                      class_path='D:/Wang_SiJia/SelfSupervised-main/Data/AID/class.txt',
                                      transform=transform)
    elif dataset == 'NWPU':
        if train == True:
            dataset = MyCustomDataset(root_path="D:/Wang_SiJia/SelfSupervised-main/Data/NWPU/train.txt",
                                      class_path='D:/Wang_SiJia/SelfSupervised-main/Data/NWPU/class.txt',
                                      transform=transform)
        else:
            dataset = MyCustomDataset(root_path="D:/Wang_SiJia/SelfSupervised-main/Data/NWPU/test.txt",
                                      class_path='D:/Wang_SiJia/SelfSupervised-main/Data/NWPU/class.txt',
                                      transform=transform)
    else:
        raise NotImplementedError

    if debug_subset_size is not None:
        dataset = torch.utils.data.Subset(dataset, range(0, debug_subset_size))  # take only one batch
        dataset.classes = dataset.dataset.classes
        dataset.targets = dataset.dataset.targets
    return dataset