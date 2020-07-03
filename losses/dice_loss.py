# Nicola Dinsdale 2020
# Dice loss for segmentation
########################################################################################################################
# Import dependencies
import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F
########################################################################################################################

class dice_loss(nn.Module):
    def __init__(self):
        super(dice_loss, self).__init__()
        self.eps=1e-7

    def forward(self, x, target):
        num_classes = target.shape[1]   # Channels first
        target = target.type(x.type())
        dims = (0,) + tuple(range(2, target.ndimension()))
        intersection = torch.sum(x * target, dims)
        cardinality = torch.sum(x + target, dims)
        dice_loss = (2. * intersection / (cardinality + self.eps)).mean()
        return (1 - dice_loss)

class dice_loss_half(nn.Module):
    def __init__(self):
        super(dice_loss_half, self).__init__()
        self.eps=1e-7

    def forward(self, x, target):
        x = x.half()
        num_classes = target.shape[1]   # Channels first
        target = target.type(x.type())
        dims = (0,) + tuple(range(2, target.ndimension()))
        intersection = torch.sum(x * target, dims)
        cardinality = torch.sum(x + target, dims)
        dice_loss = (2. * intersection / (cardinality + self.eps)).mean()
        return (1 - dice_loss)

class dice_loss_DANN(nn.Module):
    def __init__(self):
        super(dice_loss_DANN, self).__init__()
        self.eps=1e-7

    def forward(self, x, target):
        label_pred = x
        label_true = target[0]
        domains = target[1]

        _, domains = torch.max(domains, dim=1)
        bool_0 = torch.eq(domains, 0)
        bool_0 = bool_0.type(torch.LongTensor).cuda()

        indexs = torch.linspace(1, len(label_true), len(label_true))
        indexs = indexs.type(torch.LongTensor).cuda()

        msked_indexs = torch.mul(indexs, bool_0)
        msked_indexs = msked_indexs.type(torch.LongTensor).cuda()

        msked_indexs = msked_indexs[msked_indexs != 0]
        msked_indexs = msked_indexs - 1

        label_pred_msk_0 = label_pred[msked_indexs]
        label_true_msk_0 = label_true[msked_indexs]

        loss_0 = self._dice(label_pred_msk_0, label_true_msk_0)

        bool_1 = torch.eq(domains, 1)
        bool_1 = bool_1.type(torch.LongTensor).cuda()

        indexs = torch.linspace(1, len(label_true), len(label_true))
        indexs = indexs.type(torch.LongTensor).cuda()

        msked_indexs = torch.mul(indexs, bool_1)
        msked_indexs = msked_indexs.type(torch.LongTensor).cuda()

        msked_indexs = msked_indexs[msked_indexs != 0]
        msked_indexs = msked_indexs - 1


        label_pred_msk_1 = label_pred[msked_indexs]
        label_true_msk_1 = label_true[msked_indexs]

        loss_1 = self._dice(label_pred_msk_1, label_true_msk_1)

        loss = loss_0 + loss_1
        return loss, [loss_0, loss_1]

    def _dice(self, x, target):
        num_classes = target.shape[1]   # Channels first
        target = target.type(x.type())
        dims = (0,) + tuple(range(2, target.ndimension()))
        intersection = torch.sum(x * target, dims)
        cardinality = torch.sum(x + target, dims)
        dice_loss = (2. * intersection / (cardinality + self.eps)).mean()
        return (1 - dice_loss)





if __name__ == '__main__':
    x = np.load('/Users/dinsdale/Documents/OASIS/data_harmonization/pytorch_DANN/losses/y_train_oasis_segmentation_one_hot_test.npy')
    print(x.shape)
    y = np.zeros_like(x)

    y = torch.Tensor(y)
    x = torch.Tensor(x)
    domains = np.array([1.0, 0.0]).reshape(1, 2)
    domains2 = np.array([0.0, 1.0]).reshape(1,2)
    domains = np.append(domains, domains2, axis=0)
    domains = torch.Tensor(domains)
    print(domains.shape)

    loss = dice_loss_DANN()
    print(loss(y, [x, domains]))
