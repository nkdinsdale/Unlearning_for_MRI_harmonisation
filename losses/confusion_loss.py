# Nicola Dinsdale 2020
# Define the loss function for the confusion part of the network
########################################################################################################################
# Import dependencies
import torch.nn as nn
import torch
import numpy as np
########################################################################################################################

class confusion_loss(nn.Module):
    def __init__(self, task=0):
        super(confusion_loss, self).__init__()
        self.task = task

    def forward(self, x, target):
        # We only care about x
        log = torch.log(x)
        log_sum = torch.sum(log, dim=1)
        normalised_log_sum = torch.div(log_sum,  x.size()[1])
        loss = torch.mul(torch.sum(normalised_log_sum, dim=0), -1)
        return loss


if __name__ == '__main__':
    a = np.array([[0.28, 0.49, 0.23], [0.35, 0.55, 0.10]]).reshape((2, 3))
    print(a)
    a = torch.Tensor(a)

    log = torch.log(a)
    log_sum = torch.sum(log, dim=1)
    normalised_log_sum = torch.div(log_sum, a.size()[1])
    loss = torch.mul(torch.sum(normalised_log_sum, dim=0), -1)
    print(loss)
