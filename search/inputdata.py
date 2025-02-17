import os

import torch
import torchvision
from PIL import Image
from torch.utils import data
from torchvision import datasets
from torchvision.transforms import transforms
from datasets.MyDataset import MyCustomDataset
from augmentations import get_aug
from datasets import get_dataset


def input_data(args):
    # 图像预处理
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # transforms.Resize(256)
    ])
    # 创建数据集
    # train_dataset = torchvision.datasets.CIFAR10("../Data", train=True, transform=trans, download=True)
    # test_dataset = torchvision.datasets.CIFAR10("../Data", train=False, transform=trans, download=True)
    # train_data = MyCustomDataset(root_path="../Data/UCM/UCM_train_list.txt",transform=trans)
    # test_data = MyCustomDataset(root_path="../Data/UCM/UCM_test_list.txt",transform=trans)
    #
    # # classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    #
    # batch_size = 1
    # # train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # # test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    #
    # train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    # test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    train_dataloader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=trans,
            train=True,
            **args.dataset_kwargs),
        shuffle=True,
        batch_size=1,
        **args.dataloader_kwargs
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=trans,
            train=False,
            **args.dataset_kwargs),
        shuffle=False,
        batch_size=1,
        **args.dataloader_kwargs
    )
    data_loaders = {
        "train": train_dataloader,
        "test": test_dataloader
    }
    return data_loaders
    #
    # # data_dir = '../../数据集/UCM_swin'  # 样本地址
    # data_dir = '../../数据集/AID_swin'  # 样本地址
    #
    # # 构建训练和验证的样本数据集，字典格
    # # os.path.join实现路径拼接
    #
    # image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])  # data_transforms也是字典格式，
    #                   for x in ['test']}
    # # 分别对训练和验证样本集构建样本加载器，还可以针对minibatch、多线程等进行针对性构建
    # dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1, shuffle=False, num_workers=0)
    #                for x in ['test']}
    # # 分别计算训练与测试的样本数，字典格式
    # dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}  # 训练与测试的样本数
    #
    # return image_datasets, dataloaders, dataset_sizes