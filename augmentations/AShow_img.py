"""效果图显示"""
import torch
from torchvision import transforms
import PIL.Image as Image
import matplotlib.pyplot as plt
from byol_aug import BYOL_transform
from Strong_Weak_aug import StrongWeakTransform
import numpy as np
def transform_convert(img_tensor, transform):
    """
    param img_tensor: tensor
    param transforms: torchvision.transforms
    """
    if 'Normalize' in str(transform):
        normal_transform = list(filter(lambda x: isinstance(
        x, transforms.Normalize), transform.transforms))
        mean = torch.tensor(
        normal_transform[0].mean, dtype=img_tensor.dtype, device=img_tensor.device)
        std = torch.tensor(
        normal_transform[0].std, dtype=img_tensor.dtype, device=img_tensor.device)
        img_tensor.mul_(std[:, None, None]).add_(mean[:, None, None])
        img_tensor = img_tensor.transpose(0, 2).transpose(
        0, 1)  # C x H x W  ---> H x W x C

    if 'ToTensor' in str(transform) or img_tensor.max() < 1:
        img_tensor = img_tensor.detach().numpy()*255

    if  isinstance(img_tensor, torch.Tensor):
        img_tensor = img_tensor.numpy()

    if img_tensor.shape[2] == 3:
        img = Image.fromarray(img_tensor.astype('uint8')).convert('RGB')
    elif img_tensor.shape[2] == 1:
        img = Image.fromarray(img_tensor.astype('uint8')).squeeze()
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_tensor.shape[2]))
    return img
if __name__ == '__main__':
    image = Image.open("E:/王思佳的文件/代码/数据集/AID Data Set/data/AID Data Set/AID_dataset/AID\Airport/airport_1.jpg")
    transform = StrongWeakTransform(224)
    input1,input2 = transform(image)
    img1 = transform_convert(input1, transform.strongtransform)
    plt.imshow(img1)
    plt.show()
    img2 = transform_convert(input2, transform.weaktransform)
    plt.imshow(img2)
    plt.show()
