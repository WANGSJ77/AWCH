import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"）

def NT_XentLoss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N, Z = z1.shape
    device = z1.device
    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)
    l_pos = torch.diag(similarity_matrix, N)
    r_pos = torch.diag(similarity_matrix, -N)
    positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)
    diag = torch.eye(2*N, dtype=torch.bool, device=device)
    diag[N:,:N] = diag[:N,N:] = diag[:N,:N]

    negatives = similarity_matrix[~diag].view(2*N, -1)

    logits = torch.cat([positives, negatives], dim=1)
    logits /= temperature

    labels = torch.zeros(2*N, device=device, dtype=torch.int64)

    loss = F.cross_entropy(logits, labels, reduction='sum')
    return loss / (2 * N)


class projection_MLP(nn.Module):
    def __init__(self, in_dim, out_dim=256):
        super().__init__()
        hidden_dim = in_dim
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
class SimCLR(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projector = projection_MLP(backbone.output_dim)
        self.encoder = nn.Sequential(
            self.backbone,
            self.projector
        )
    def forward(self, x1, x2,flag=8):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        if flag==0:
            loss = NT_XentLoss2(z1, z2)
            return {'loss':loss}
        elif flag == 1:
            loss = criterion(z1,z2)
            return {'loss':loss}
        elif flag == 2:
            f = self.encoder
            feat_high = f[0](x1)
            feat_low = f[1](feat_high)
            return feat_high, feat_low
        elif flag == 3:
            loss = NT_XentLoss_Gaussian_neg(z1, z2,0.5)
            print("去噪：{}，温度系数:{}".format(loss,0.5))
            return {'loss': loss}