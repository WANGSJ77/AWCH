import torchvision.models
from .simsiam import SimSiam
from .byol import BYOL
from .simclr import SimCLR
import torch
from .simclr_Hash import SimCLR_Hash
from models.backbones.resnet import resnet50
# from models.backbones.resnet_selfatt import resnet50
# from torchvision.models import resnet18,resnet50
# from .backbones import resnet18_cifar_variant1, resnet18_cifar_variant2
import numpy as np

def get_backbone(backbone, castrate=True):
    backbone = eval(f"{backbone}(pretrained=True)")
    # if flag == False:
    #     if backbone == 'resnet18':
    #         backbone = resnet18_cifar_variant1(pretrained=True)
    #     elif backbone == 'resnet50':
    #         backbone = resnet18_cifar_variant1(pretrained=True)
    # else:
    #     # backbone = eval(f"{backbone}()")
    #     if backbone == 'resnet18':
    #         backbone = resnet18(pretrained=True)
    #     elif backbone == 'resnet50':
    #         backbone = resnet50(pretrained=True)
    if castrate:
        backbone.output_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()

    return backbone


def get_model(model_cfg):    

    if model_cfg.name == 'simsiam':
        model =  SimSiam(get_backbone(model_cfg.backbone))
        if model_cfg.proj_layers is not None:
            model.projector.set_layers(model_cfg.proj_layers)
    elif model_cfg.name == 'byol':
        model = BYOL(get_backbone(model_cfg.backbone))
    elif model_cfg.name == 'simclr':
        model = SimCLR(get_backbone(model_cfg.backbone))
    elif model_cfg.name == 'simclr_hash':
        model = SimCLR_Hash(get_backbone(model_cfg.backbone))
    elif model_cfg.name == 'swav':
        raise NotImplementedError
    else:
        raise NotImplementedError
    return model






