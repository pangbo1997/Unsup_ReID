import torch
import torch.cuda
import torch.nn.functional as F
import numpy as np
import random
class MemoryBank(object):
    def __init__(self,bank_size,device,cameras,positive_num,negative_num ,lambda_g, lambda_c):
        self.cameras=np.array(cameras)

        self.lambda_g = lambda_g
        self.lambda_c = lambda_c
        self.bank_size=bank_size
        self.positive_num = positive_num
        self.negative_num= min(negative_num,self.bank_size[0]-self.positive_num-1)

        self.global_vectors=torch.zeros(bank_size,requires_grad=False)
        self.local_vectors=torch.zeros(8,bank_size[0],bank_size[1]//8,requires_grad=False)
        self.device=device

    def init_vector(self,model,train_loader):
        model=model.to(self.device)
        with torch.no_grad():
            for img,mem_idx in train_loader:
                img=img.to(self.device)
                global_feats,local_feats=model(img)
                self.global_vectors[mem_idx]=global_feats.detach().cpu()
                self.local_vectors[:,mem_idx,:]=local_feats.detach().cpu()


    def cross_camera_encouragment(self, idx):
        #print(self.lambda_c)
        cameras=self.cameras[idx]
        cce = np.zeros((len(cameras), len(self.cameras)))
        for i, camera in enumerate(cameras):
            truth_vector = self.cameras == camera
            cce[i] = self.lambda_c*truth_vector
        cce=torch.tensor(cce,dtype=torch.float32)
        return cce


    def update(self, index,gloabl_feats,local_feats):
        self.global_vectors[index] = 0.5*self.global_vectors[index]+0.5*gloabl_feats.detach().cpu()
        self.global_vectors[index]=  F.normalize(self.global_vectors[index],2)

        self.local_vectors[:,index,:]=0.5*self.local_vectors[:,index,:]+0.5*local_feats.detach().cpu()
        self.local_vectors[:,index,:]=F.normalize(self.local_vectors[:,index,:],p=2,dim=2)


    def cal_distance(self,qf,gf):
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        return  distmat

    def global_distance(self,global_feats):
        return self.cal_distance(global_feats.detach().cpu(),self.global_vectors)

    def local_distance(self,local_feats):
        local_distance=0
        for i in range(8):
            local_distance=local_distance+self.cal_distance(local_feats[i].detach().cpu(),self.local_vectors[i]
                                                      )
        return local_distance/8

    def get_soft_label(self,index,global_feats,local_feats,init=False):
        #distance = (1 - self.lambda_p)*self.global_distance(global_feats)+self.lambda_p*self.local_distance(local_feats)+self.cce[index]
        if init:
            positive_idx=index.reshape(len(index), 1)

            negative_num=self.negative_num+self.positive_num

            negative_idx = torch.zeros(len(index), negative_num, dtype=torch.int64).to(self.device)

            re_idx=np.arange(self.bank_size[0])
            mem_idx=index.detach().cpu().numpy()
            random.shuffle(re_idx)
            for i in range(len(index)):
                negative_idx[i]=torch.tensor(np.delete(re_idx,mem_idx[i])[:negative_num]).to(self.device)
            return positive_idx,negative_idx
        else:
            distance = self.lambda_g * self.global_distance(global_feats) + (1-self.lambda_g) * self.local_distance(
                local_feats) + self.cross_camera_encouragment(index.detach().cpu())

            re_idx= distance.argsort(dim=1).to(self.device)

            positive_idx=torch.zeros(len(index),self.positive_num+1,dtype=torch.int64).to(self.device)
            negative_idx=torch.zeros(len(index),self.negative_num,dtype=torch.int64).to(self.device)

            positive_idx[:,0]=index
            for i in range(len(index)):
                keep=1-(index[i]==re_idx[i])
                kidx=re_idx[i][keep]
                positive_idx[i,1:self.positive_num+1]=kidx[0:self.positive_num]
                negative_idx[i]=kidx[self.positive_num:self.negative_num+self.positive_num]

        return positive_idx,negative_idx





