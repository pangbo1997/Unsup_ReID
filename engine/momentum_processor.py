import torch
import torch.nn as nn
from modeling import build_model
import random
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import os.path as osp
import time
import torchvision.transforms as T
from data.datasets.dataset_loader import ImageDataset


def train_collate_fn(batch):
    imgs, _, _, _, mem_idx= zip(*batch)
    mem_idx = torch.tensor(mem_idx, dtype=torch.int64)
    return torch.stack(imgs, dim=0), mem_idx


class Queue():
    def __init__(self,embeding_fea_size,device):
        self.max_num=5000

        self.index=torch.zeros(self.max_num).to(device)
        self.feats=torch.zeros(self.max_num,embeding_fea_size).to(device)

        self.ptr=0
        self.qtr=0

        self.count=0
    def enqueue(self,idx,feat):
        while idx==self.index[self.qtr]:
            self.qtr = (self.qtr + 1) % self.max_num

        self.index[self.qtr]=idx
        self.feats[self.qtr]=feat

        self.qtr=(self.qtr+1)%self.max_num




class MoProc:
    def __init__(self,cfg,model,device,trainset,lambda_c):

        self.mo_encoder=build_model(cfg)
        self.cfg=cfg
        self.device=device
        self.mo_encoder=self.mo_encoder.to(self.device)
        self.cameras=np.array(trainset.camids)
        self.m=0.999
        self.lambda_c=lambda_c

        self.positive_num=cfg.MODEL.POSITIVE_NUM
        self.negative_num=cfg.MODEL.NEGATIVE_NUM
        self.neighbour_num=cfg.MODEL.POSITIVE_NUM+cfg.MODEL.NEGATIVE_NUM

        self.dataset=trainset

        self.dataloader= DataLoader(
            trainset, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS,
            collate_fn=train_collate_fn
        )

        self.init_cross_camera_encouragment()
        self.init_mo_encoder(model)
        self.update_neighbour()

    def init_mo_encoder(self,model):
        for param,param_m in zip(model.parameters(),self.mo_encoder.parameters()):
            param_m.data.copy_(param.data)
            param_m.requires_grad=False

    def init_cross_camera_encouragment(self):

        cameras=self.cameras
        cce = np.zeros((len(cameras), len(self.cameras)))
        for i, camera in enumerate(cameras):
            truth_vector = self.cameras == camera
            cce[i] = self.lambda_c*truth_vector
        self.cce=torch.tensor(cce,dtype=torch.float32)

    def update_mo_encoder(self,model):
        for param, param_m in zip(model.parameters(), self.mo_encoder.parameters()):
            param_m.data = param_m.data * self.m + param.data * (1. - self.m)

    def update_neighbour(self):
        print('Update MoProc neightbour info')
        start=time.time()
        model=self.mo_encoder
        global_vectors=torch.zeros(len(self.dataset),self.cfg.MODEL.EMBEDING_FEA_SIZE,requires_grad=False)
        local_vectors=torch.zeros(8,len(self.dataset),self.cfg.MODEL.EMBEDING_FEA_SIZE//8,requires_grad=False)
        with torch.no_grad():
            for img,mem_idx in self.dataloader:
                img=img.to(self.device)
                global_feats,local_feats=model(img)
                global_vectors[mem_idx]=global_feats.detach().cpu()
                local_vectors[:,mem_idx,:]=local_feats.detach().cpu()

        global_distmat=self.cal_distmat(global_vectors,global_vectors)
        local_distmat=0
        for i in range(8):
                local_distmat+=self.cal_distmat(local_vectors[i],local_vectors[i])

        distmat=global_distmat+local_distmat/8+self.cce
        distmat=distmat.detach().cpu().numpy()
        indices = np.argsort(distmat, axis=1)

        self.neighbour_idx=indices[:,:self.neighbour_num+1]
        end=time.time()
        print('Update Done,Running time: %s Seconds' % (end - start))


    def cal_distmat(self,feats_1,feats_2=None):
        m,n=feats_1.shape[0],feats_2.shape[0]
        distmat = torch.pow(feats_1, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(feats_2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, feats_1, feats_2.t())
        return distmat




    def cross_camera_encouragment(self,cameras,neighbour_camera):

        truth_vector = cameras == neighbour_camera
        return self.lambda_c*truth_vector


    # def get_soft_label(self,index,init=False):
    #     if init:
    #         positive_idx=index.reshape(len(index), 1)
    #
    #         negative_num=self.negative_num+self.positive_num
    #
    #         negative_idx = torch.zeros(len(index), negative_num, dtype=torch.int64).to(self.device)
    #
    #         re_idx=np.arange(len(self.dataset))
    #         mem_idx=index.detach().cpu().numpy()
    #         random.shuffle(re_idx)
    #         for i in range(len(index)):
    #             negative_idx[i]=torch.tensor(np.delete(re_idx,mem_idx[i])[:negative_num]).to(self.device)
    #         return positive_idx,negative_idx
    #     else:
    #                 positive_idx=torch.zeros(len(index),self.positive_num+1,dtype=torch.int64).to(self.device)
    #                 negative_idx=torch.zeros(len(index),self.negative_num,dtype=torch.int64).to(self.device)
    #
    #                 mem_idx=index.detach().cpu().numpy()
    #                 positive_idx[:,0]=index
    #                 for i in range(len(index)):
    #                     with torch.no_grad():
    #                         neighbour_img=torch.stack([self.dataset[j][0] for j in self.neighbour_idx[mem_idx[i]]],dim=0).to(self.device)
    #                         neighbour_feat_global,neighbour_feat_local=self.mo_encoder(neighbour_img)
    #                     global_distmat=self.cal_distmat(global_feats[i:i+1],neighbour_feat_global)
    #                     local_distmat=0
    #                     for j in range(8):
    #                         local_distmat += self.cal_distmat(local_feats[j,i:i+1],neighbour_feat_local[j])
    #
    #
    #                     distmat=global_distmat+local_distmat/8
    #                     distmat=distmat.detach().cpu().numpy()
    #                     distmat+=self.cross_camera_encouragment(self.cameras[index[i]],self.cameras[self.neighbour_idx[index[i]]])
    #
    #                     re_idx=torch.tensor(self.neighbour_idx[mem_idx[i]][np.argsort(distmat)[0]]).to(self.device)
    #
    #                     keep=1-(index[i]==re_idx)
    #                     kidx=re_idx[keep]
    #                     positive_idx[i,1:self.positive_num+1]=kidx[0:self.positive_num]
    #                     negative_idx[i]=kidx[self.positive_num:self.negative_num+self.positive_num]
    #
    #             return positive_idx,negative_idx




    def get_soft_label(self,index,global_feats,local_feats,init=False):
        if init:
            # positive_idx=index.reshape(len(index), 1)
            # negative_idx = torch.zeros(len(index), self.neighbour_num, dtype=torch.int64).to(self.device)
            # mem_idx=index.detach().cpu().numpy()
            # for i in range(len(index)):
            #     negative_idx[i]=torch.tensor(np.delete(self.neighbour_idx[mem_idx[i]],mem_idx[i])[:self.neighbour_num]).to(self.device)
            # return positive_idx,negative_idx
            positive_idx=index.reshape(len(index), 1)

            negative_num=self.negative_num+self.positive_num

            negative_idx = torch.zeros(len(index), negative_num, dtype=torch.int64).to(self.device)

            re_idx=np.arange(len(self.dataset))
            mem_idx=index.detach().cpu().numpy()
            random.shuffle(re_idx)
            for i in range(len(index)):
                negative_idx[i]=torch.tensor(np.delete(re_idx,mem_idx[i])[:negative_num]).to(self.device)
            return positive_idx,negative_idx
        else:
            positive_idx=torch.zeros(len(index),self.positive_num+1,dtype=torch.int64).to(self.device)
            negative_idx=torch.zeros(len(index),self.negative_num,dtype=torch.int64).to(self.device)

            mem_idx=index.detach().cpu().numpy()
            positive_idx[:,0]=index
            for i in range(len(index)):
                with torch.no_grad():
                    neighbour_img=torch.stack([self.dataset[j][0] for j in self.neighbour_idx[mem_idx[i]]],dim=0).to(self.device)
                    neighbour_feat_global,neighbour_feat_local=self.mo_encoder(neighbour_img)
                global_distmat=self.cal_distmat(global_feats[i:i+1],neighbour_feat_global)
                local_distmat=0
                for j in range(8):
                    local_distmat += self.cal_distmat(local_feats[j,i:i+1],neighbour_feat_local[j])


                distmat=global_distmat+local_distmat/8
                distmat=distmat.detach().cpu().numpy()
                distmat+=self.cross_camera_encouragment(self.cameras[index[i]],self.cameras[self.neighbour_idx[index[i]]])

                re_idx=torch.tensor(self.neighbour_idx[mem_idx[i]][np.argsort(distmat)[0]]).to(self.device)

                keep=1-(index[i]==re_idx)
                kidx=re_idx[keep]
                positive_idx[i,1:self.positive_num+1]=kidx[0:self.positive_num]
                negative_idx[i]=kidx[self.positive_num:self.negative_num+self.positive_num]

        return positive_idx,negative_idx







