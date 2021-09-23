# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .baseline import Baseline
from .end2end import End2End_AvgPooling

def build_model(cfg, num_classes):
    # if cfg.MODEL.NAME == 'resnet50':
    #     model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT)
    #model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE)
    model =End2End_AvgPooling(dropout=0.5,embeding_fea_size=cfg.MODEL.EMBEDING_FEA_SIZE,is_video=cfg.DATASETS.IS_VIDEO,local_conv_out_channels=cfg.MODEL.EMBEDING_FEA_SIZE//8)
    return model
