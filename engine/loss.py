from __future__ import absolute_import

import torch
import torch.nn.functional as F
from torch import nn, autograd
import numpy as np


class Exclusive(autograd.Function):
    def __init__(self, M):
        super(Exclusive, self).__init__()
        self.M = M

    def forward(self, inputs, index,positive_index):
        self.save_for_backward(inputs, index,positive_index)

        N = inputs.size(0)
        K=  index.size(1)

        logits = torch.zeros(N,K).to(inputs.device)
        for i in range(N):
            logits[i] = inputs[i:i + 1].mm(self.M[index[i]].t())
        #logits=inputs.mm(self.M.t())
        return logits

    def backward(self, grad_outputs):
        inputs, index, positive_index = self.saved_tensors
        N,C=inputs.size()
        grad_inputs=torch.zeros(N,C).to(inputs.device)

        for i in range(N):
            grad_inputs[i] = grad_outputs[i:i + 1].mm(self.M[index[i]])


        # print(targets)
        for x, y in zip(inputs, positive_index):
            self.M[y] = F.normalize((self.M[y] + x) / 2, p=2, dim=0)

        return grad_inputs, None, None




class ExLoss(nn.Module):
    def __init__(self, num_features, num_classes, t=1.0,
                 weight=None):
        super(ExLoss, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.t = t
        self.weight = weight
        self.index_list = np.arange(num_classes)

        self.register_buffer('M', torch.zeros(num_classes, num_features))

    def forward(self, inputs, positive_index,negative_index, cof):

        index_list=torch.cat((positive_index,negative_index),dim=1)


        loss = 0
        N,K=positive_index.size()

        for j in range(positive_index.size(1)):
            logits=Exclusive(self.M)(inputs,index_list,index_list[:,j])*self.t
            loss += cof[j] * F.cross_entropy(logits, j + torch.zeros(N, dtype=torch.int64).to(device=inputs.device),
                                             weight=self.weight)
            #loss += cof[j] * F.cross_entropy(logits, targets[:,j],weight=self.weight)
        return loss, logits




def loss_fn(criterion, inputs,positive_index,negative_index, lambda_t, positive_num, alpha):

    cof = np.zeros(positive_num+1)
    cof[0] = lambda_t
    cof[1:] = alpha * (1 - lambda_t) / positive_num
    loss, outputs = criterion(inputs, positive_index,negative_index, cof)


    return loss, outputs
