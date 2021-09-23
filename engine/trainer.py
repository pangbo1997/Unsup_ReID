# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging

import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage
import numpy as np
from utils.reid_metric import R1_mAP
global ITER
ITER = 0
from .loss import *
from .memory_bank import MemoryBank
def create_supervised_trainer(model, optimizer,memory_bank, criterion,lambda_t,lambda_p,alpha,
                              device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)



    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        img, mem_idx = batch

        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        mem_idx=mem_idx.to(device)
        global_feats,local_feats = model(img)

        if engine.state.epoch<=3:
            positive_index,negative_index= memory_bank.get_soft_label(mem_idx,global_feats,local_feats,init=True)
        else:
            positive_index,negative_index= memory_bank.get_soft_label(mem_idx,global_feats,local_feats)

        loss_g,score_g=loss_fn(criterion,global_feats,positive_index,negative_index,lambda_t=lambda_t,positive_num=memory_bank.positive_num,alpha=alpha)
        loss_l,score_l=loss_fn(criterion,torch.cat(list(local_feats),dim=1),positive_index,negative_index,lambda_t=lambda_t,positive_num=memory_bank.positive_num,alpha=alpha)
        loss=lambda_p*loss_l+(1-lambda_p)*loss_g


        loss.backward()
        optimizer.step()

        memory_bank.update(mem_idx,global_feats,local_feats)
        #criterion.update(targets,global_feats,torch.cat(list(local_feats),dim=1))


        # compute acc
        acc_g = (score_g.max(1)[1] == torch.zeros_like(mem_idx)).float().mean()
        acc_l=(score_l.max(1)[1] ==torch.zeros_like(mem_idx)).float().mean()
        # acc_g = (score_g.max(1)[1] == mem_idx).float().mean()
        # acc_l=(score_l.max(1)[1] ==mem_idx).float().mean()
        return loss.item(),acc_g.item(),acc_l.item()


    return Engine(_update)


def create_supervised_trainer_with_center(model, center_criterion, optimizer, optimizer_center, loss_fn, cetner_loss_weight,
                              device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        optimizer_center.zero_grad()
        img, target = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        score, feat = model(img)
        loss = loss_fn(score, feat, target)
        # print("Total loss is {}, center loss is {}".format(loss, center_criterion(feat, target)))
        loss.backward()
        optimizer.step()
        for param in center_criterion.parameters():
            param.grad.data *= (1. / cetner_loss_weight)
        optimizer_center.step()

        # compute acc
        acc = (score.max(1)[1] == target).float().mean()
        return loss.item(), acc.item()

    return Engine(_update)


def create_supervised_evaluator(model, metrics,
                                device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            gloabl_feat,local_feat = model(data)
            gloabl_feat=gloabl_feat.detach().cpu()
            local_feat=local_feat.detach().cpu()
            return gloabl_feat,local_feat, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        num_query,
        start_epoch
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")
    print('lambda_t:{:.4},lambda_p:{:.4},lambda_c:{:.4},positive_num:{},negative_num:{},lambda_g:{:.4},alpha:{:.4}'.format(cfg.MODEL.LAMBDA_T,cfg.MODEL.LAMBDA_P,cfg.MODEL.LAMBDA_C,cfg.MODEL.POSITIVE_NUM,cfg.MODEL.NEGATIVE_NUM,cfg.MODEL.LAMBDA_G,cfg.MODEL.ALPHA))

    num_classes=len(train_loader.dataset)
    memory_bank=MemoryBank((num_classes,cfg.MODEL.EMBEDING_FEA_SIZE),device,train_loader.dataset.camids,positive_num=cfg.MODEL.POSITIVE_NUM,negative_num=cfg.MODEL.NEGATIVE_NUM,lambda_g=cfg.MODEL.LAMBDA_G,lambda_c=cfg.MODEL.LAMBDA_C)
    memory_bank.init_vector(model,train_loader)
    criterion = ExLoss(cfg.MODEL.EMBEDING_FEA_SIZE, num_classes, t=10).to(device)
   # print('lambda_t:{:.4},lambda_p:{:.4},lambda_c:{:.4},reliable_num:{},lambda_g:{:.4},alpha:{:.4}'.format(cfg.MODEL.LAMBDA_T,cfg.MODEL.LAMBDA_P,cfg.MODEL.LAMBDA_C,cfg.MODEL.RELIABLE_NUM,cfg.MODEL.LAMBDA_G,cfg.MODEL.ALPHA))
    trainer = create_supervised_trainer(model, optimizer,memory_bank,criterion, lambda_t=cfg.MODEL.LAMBDA_T,lambda_p=cfg.MODEL.LAMBDA_P,alpha=cfg.MODEL.ALPHA,device=device)
    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, lambda_g=cfg.MODEL.LAMBDA_G,lambda_c=cfg.MODEL.LAMBDA_C,max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=20, require_empty=False,save_as_state_dict=True)
    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
                                                                     'optimizer': optimizer})
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc_g')
    RunningAverage(output_transform=lambda x: x[2]).attach(trainer, 'avg_acc_l')
    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1
        if ITER % log_period == 0:
            #print((criterion.M - criterion.V).sum())
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc_g: {:.3f},Acc_l: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, ITER, len(train_loader),
                                engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc_g'],engine.state.metrics['avg_acc_l'],
                                scheduler.get_lr()[0]))
        if len(train_loader) == ITER:
            ITER = 0

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    trainer.run(train_loader, max_epochs=epochs)


def do_train_with_center(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_fn,
        num_query,
        start_epoch
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")
    trainer = create_supervised_trainer_with_center(model, center_criterion, optimizer, optimizer_center, loss_fn, cfg.SOLVER.CENTER_LOSS_WEIGHT, device=device)
    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=10, require_empty=False)
    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
                                                                     'optimizer': optimizer,
                                                                     'center_param': center_criterion,
                                                                     'optimizer_center': optimizer_center})

    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1

        if ITER % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, ITER, len(train_loader),
                                engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc'],
                                scheduler.get_lr()[0]))
        if len(train_loader) == ITER:
            ITER = 0

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    trainer.run(train_loader, max_epochs=epochs)
