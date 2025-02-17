import os
import scipy.io as sio
import numpy as np
import torch
import torchvision
from torch import nn
from torchvision import datasets
from torchvision.transforms import transforms
from tqdm import tqdm
from arguments import get_args
from augmentations import get_aug
from datasets import get_dataset
from models import get_model, SimSiam, get_backbone
from search.inputdata import input_data
# --data_dir ../Data/ --log_dir ../logs/ -c ../configs/simclr_ucm.yaml --ckpt_dir ../cache/ --hide_progress
from models.backbones.resnet import resnet50
def gettest(device, args):
    model = get_model(args.model)
    weights_path = '../cache/simclr-UCM-resnet50_0802202754.pth'
    weights = torch.load(weights_path)['state_dict']  # 读取训练模型权重
    model.load_state_dict(weights)
    model = model.to(device)
    model.eval()
    # model = torchvision.models.resnet18(pretrained=True)
    features_h = []
    features_l = []
    lables = []
    dataloaders = input_data(args=args)
    # image_datasets, dataloaders, dataset_sizes = input_data(args=args)
    with torch.no_grad():
        for image, lable in tqdm(dataloaders['test']):
            # for data in tqdm(dataloaders['test']):
            #     torch.cuda.empty_cache()
            #     image, lable = data
            image = image.to(device)
            lable = lable.to(device)
            feature = model(image,image,flag=3)
            feature_high = feature[0].view(-1)
            feature_low = feature[1].view(-1)
            lable = lable.view(-1)
            feature_high = feature_high.cpu().numpy()
            feature_low = feature_low.cpu().numpy()
            lable = lable.cpu().numpy()
            features_h.append(feature_high)
            features_l.append(feature_low)
            lables.append(lable)
    data = {'features':features_h,'labels':lables}
    sio.savemat('../特征和标签/feature_test.mat', data)
def gettrain(device, args):
    model = get_model(args.model)
    weights_path = '../cache/simclr-UCM-resnet50_0802202754.pth'
    weights = torch.load(weights_path)['state_dict']  # 读取训练模型权重
    model.load_state_dict(weights)
    model = model.to(device)
    model.eval()
    # model = torchvision.models.resnet18(pretrained=True)
    features_h = []
    features_l = []
    lables = []
    dataloaders = input_data(args=args)
    # image_datasets, dataloaders, dataset_sizes = input_data(args=args)
    with torch.no_grad():
        for image, lable in tqdm(dataloaders['train']):
            # for data in tqdm(dataloaders['test']):
            #     torch.cuda.empty_cache()
            #     image, lable = data
            image = image.to(device)
            lable = lable.to(device)
            feature = model(image,image,flag=3)
            feature_high = feature[0].view(-1)
            feature_low = feature[1].view(-1)
            lable = lable.view(-1)
            feature_high = feature_high.cpu().numpy()
            feature_low = feature_low.cpu().numpy()
            lable = lable.cpu().numpy()
            features_h.append(feature_high)
            features_l.append(feature_low)
            lables.append(lable)
    data = {'features': features_h, 'labels': lables}
    sio.savemat('../特征和标签/feature_train.mat', data)
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gettrain(device=device, args=get_args())
    gettest(device=device, args=get_args())