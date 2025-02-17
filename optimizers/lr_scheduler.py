import torch
import numpy as np
# from torch.optim.lr_scheduler import _LRScheduler
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class LR_Scheduler(object):
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch, constant_predictor_lr=False):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr+0.5*(base_lr-final_lr)*(1+np.cos(np.pi*np.arange(decay_iter)/decay_iter))
        
        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0
    def step(self):
        for param_group in self.optimizer.param_groups:

            if self.constant_predictor_lr and param_group['name'] == 'predictor':
                param_group['lr'] = self.base_lr
            else:
                lr = param_group['lr'] = self.lr_schedule[self.iter]
        
        self.iter += 1
        self.current_lr = lr
        return lr
    def get_lr(self):
        return self.current_lr

# optimizer:优化器，用于更新模型的参数。
# warmup_epochs: 在训练过程中逐渐增加学习率的步骤，通常称为"学习率预热"。这个参数指定了预热的时长，即在这些epoch中，学习率将从一个较小的值逐渐增加到base_lr。
# warmup_lr: 预热期间学习率的初始值。通常情况下，预热的学习率要比base_lr小得多，以避免在训练初期对模型参数进行过大的更新。
# num_epochs: 总的训练时长，通常用epoch表示。
# base_lr: 训练的初始学习率。学习率控制了参数更新的步长，太小会导致训练收敛速度过慢，而太大则可能导致训练过程不稳定。
# final_lr: 训练结束时的学习率。通常情况下，学习率会随着训练的进行逐渐减小，最终趋于一个很小的值，以保证模型能够收敛到最优解。
# iter_per_epoch: 每个epoch中的迭代次数，通常等于数据集大小除以batch size。
if __name__ == "__main__":
    import torchvision
    model = torchvision.models.resnet50()
    optimizer = torch.optim.SGD(model.parameters(), lr=999)
    epochs = 200
    n_iter = 1000
    scheduler = LR_Scheduler(optimizer, 10, 0.003, epochs, 0.03, 0, n_iter)
    import matplotlib.pyplot as plt
    lrs = []
    for epoch in range(epochs):
        for it in range(n_iter):
            lr = scheduler.step()
            lrs.append(lr)
    plt.plot(lrs)
    plt.show()