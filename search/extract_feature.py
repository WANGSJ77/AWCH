import os
from search.evaluation import evaluation
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

def main(device, args):
    model = get_model(args.model)
    weights_path = '../cache/simclr-UCM-resnet50_240830175901.pth'
    weights = torch.load(weights_path)['state_dict']  # 读取训练模型权重
    model.load_state_dict(weights)
    model = model.to(device)
    model.eval()
    features_h = []
    features_l = []
    lables = []
    hashcode = []
    dataloaders = input_data(args=args)
    # image_datasets, dataloaders, dataset_sizes = input_data(args=args)
    with torch.no_grad():
        for image, lable in tqdm(dataloaders['test']):
            # for data in tqdm(dataloaders['test']):
            #     torch.cuda.empty_cache()
            #     image, lable = data
            image = image.to(device)
            lable = lable.to(device)
            feature = model(image, image, flag=2)
            feature_high = feature[0].view(-1)
            feature_low = feature[1].view(-1)
            lable = lable.view(-1)
            feature_high = feature_high.cpu().numpy()
            feature_low = feature_low.cpu().numpy()
            lable = lable.cpu().numpy()
            features_h.append(feature_high)
            features_l.append(feature_low)
            lables.append(lable)
            # code = model(image, image, flag=3)
            # code = code.cpu().numpy()
            # hashcode.append(code)
    high_feature = torch.Tensor(features_h)
    high_feature = torch.squeeze(high_feature)
    high_feature = np.array(high_feature)
    low_feature = torch.Tensor(features_l)
    low_feature = torch.squeeze(low_feature)
    low_feature = np.array(low_feature)
    # hashcode = torch.Tensor(hashcode)
    # hashcode = torch.squeeze(hashcode)
    # hashcode = np.array(hashcode)
    lables = np.array(lables)
    print("\n-------------high------------------------------")
    evaluation(high_feature, lables, "euclidean_distance")
    print("\n-------------low------------------------------")
    evaluation(low_feature, lables, "euclidean_distance")
    # print("\n-------------hash------------------------------")
    # evaluation(hashcode, lables, "euclidean_distance")
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(device=device, args=get_args())