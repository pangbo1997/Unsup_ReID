from __future__ import absolute_import

from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn import init
import torch
import torchvision
import math

from .resnet import *


__all__ = ["End2End_AvgPooling"]


class AvgPooling(nn.Module):
    def __init__(self, input_feature_size, embeding_fea_size=1024, dropout=0.5):
        super(self.__class__, self).__init__()

        # embeding
        self.embeding_fea_size = embeding_fea_size
        self.embeding = nn.Linear(input_feature_size, embeding_fea_size)
        self.embeding_bn = nn.BatchNorm1d(embeding_fea_size)
        init.kaiming_normal_(self.embeding.weight, mode='fan_out')
        init.constant_(self.embeding.bias, 0)
        init.constant_(self.embeding_bn.weight, 1)
        init.constant_(self.embeding_bn.bias, 0)
        self.drop = nn.Dropout(dropout)

    def forward(self, net):
        net =F.avg_pool2d(net,net.size()[2:]).view(net.size()[0:2],-1)
        net = self.embeding(net)
        net = self.embeding_bn(net)
        net = F.normalize(net, p=2, dim=1)
        return net
class LocalPooling(nn.Module):
    def __init__(
            self,
            num_stripes=8,
            local_conv_out_channels=256,
    ):
        super(LocalPooling, self).__init__()

        self.num_stripes = num_stripes
        self.local_conv_out_channels=local_conv_out_channels
        self.local_conv_list = nn.ModuleList()
        for _ in range(num_stripes):
            self.local_conv_list.append(nn.Sequential(
                nn.Conv2d(2048, local_conv_out_channels, 1),
                nn.BatchNorm2d(local_conv_out_channels)
            ))


    def forward(self, feat):
        """
        Returns:
          local_feat_list: each member with shape [N, c]
          logits_list: each member with shape [N, num_classes]
        """
        # shape [N, C, H, W]

        stripe_h = int(feat.size(2) / self.num_stripes)
        local_feat_list = torch.zeros(self.num_stripes,feat.size(0),self.local_conv_out_channels,device=feat.device)
        for i in range(self.num_stripes):
            # shape [N, C, 1, 1]
            local_feat = F.avg_pool2d(
                feat[:, :, i * stripe_h: (i + 1) * stripe_h, :],
                (stripe_h, feat.size(-1)))
            # shape [N, c, 1, 1]
            local_feat = self.local_conv_list[i](local_feat)

            local_feat=F.normalize(local_feat,p=2,dim=1)
            # shape [N, c]
            local_feat_list[i] = local_feat.view(local_feat.size(0), -1)


        return local_feat_list
class End2End_AvgPooling(nn.Module):

    def __init__(self, dropout=0,  embeding_fea_size=2048, fixed_layer=True,num_stripes=8,local_conv_out_channels=256):
        super(self.__class__, self).__init__()
        self.CNN = resnet50(pretrained=True,last_conv_stride=1,last_conv_dilation=1)
        self.avg_pooling = AvgPooling(input_feature_size=2048, embeding_fea_size = embeding_fea_size, dropout=dropout)
        self.local_pooling=LocalPooling(num_stripes=num_stripes,local_conv_out_channels=local_conv_out_channels)

    def forward(self, x):
        resnet_feature = self.CNN(x)
        if self.training:
            global_fea = self.avg_pooling(resnet_feature)
            local_fea=self.local_pooling(resnet_feature)
            return global_fea,local_fea
        else:
            global_fea = self.avg_pooling(resnet_feature)
            #return global_fea
            local_fea = self.local_pooling(resnet_feature)
            return global_fea,torch.cat(list(local_fea), dim=1)
        #return output


