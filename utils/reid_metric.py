# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import numpy as np
import torch
from ignite.metrics import Metric

from data.datasets.eval_reid import eval_func
from .re_ranking import re_ranking


class R1_mAP(Metric):
    def __init__(self, num_query, lambda_g,lambda_c,max_rank=50, feat_norm='yes'):
        super(R1_mAP, self).__init__()
        #print(lambda_p)
        self.num_query = num_query
        self.lambda_g=lambda_g
        self.lambda_c=lambda_c
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.global_feats = []
        self.local_feats=[]
        self.pids = []
        self.camids = []

    def update(self, output):
        global_feats,local_feats, pid, camid = output
        self.global_feats.append(global_feats)
        self.local_feats.append(local_feats)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        cce=np.zeros((len(q_camids),len(g_camids)))
#        np.save('q.npy',q_camids)
#        np.save('g.npy',g_camids)
        for i in range(self.num_query):
            cce[i]=self.lambda_c*(q_camids[i]==g_camids)
#        np.save('cce.npy',cce)
        cce=torch.tensor(cce,dtype=torch.float32)
        #def cal_distance(feats):

            #feats = torch.cat(feats, dim=0)
            #if self.feat_norm == 'yes':
            #    print("The test feature is normalized")
            #    feats = torch.nn.functional.normalize(feats, dim=1, p=2)
            # query
            #qf = feats[:self.num_query]
            # gallery
            #gf = feats[self.num_query:]

            #m, n = qf.shape[0], gf.shape[0]
            #distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                      #torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            #distmat.addmm_(1, -2, qf, gf.t())
            #return  distmat
        self.global_feats = torch.cat(self.global_feats, dim=0)
        self.local_feats=torch.cat(self.local_feats,dim=1)
        global_distance=torch.cdist(self.global_feats[:self.num_query],self.global_feats[self.num_query:])
        local_distance=0
        for i in range(8):
            local_distance=local_distance+torch.cdist(self.local_feats[i,:self.num_query].detach().cpu(),self.local_feats[i,self.num_query:]
                                                      )
        local_distance/=8
        distmat=self.lambda_g*global_distance+(1-self.lambda_g)*local_distance+cce
        #distmat = (1-self.lambda_p)*cal_distance(self.global_feats)+self.lambda_p*cal_distance(self.local_feats)
        #distmat = cal_distance(self.global_feats)# + 0.5 * cal_distance(self.local_feats)
        distmat=distmat.cpu().numpy()
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP


class R1_mAP_reranking(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP_reranking, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)

        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        # m, n = qf.shape[0], gf.shape[0]
        # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # distmat.addmm_(1, -2, qf, gf.t())
        # distmat = distmat.cpu().numpy()
        print("Enter reranking")
        distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP
