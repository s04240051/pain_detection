import numpy as np
import os
import torch
import torch.nn as nn 
import torch.backends.cudnn as cudnn

#import _init_path
from lib.model.two_stream import two_stream_model
from lib.data_build import Data_loader
from lib.config_file import cfg
import lib.model.utils as utils

logger = utils.get_logger(__name__)
def build_model(cfg):
    if cfg.MODEL.MODEL_NAME == "two_lstm":
        model = two_stream_model(cfg).cuda()
    if cfg.SOLVER.METHOD == "sgd":
        optim = torch.optim.SGD(
            model.parameters(), 
            lr=cfg.SOLVER.BASE_LR, 
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            dampening=cfg.SOLVER.DAMPENING,
            nesterov=cfg.SOLVER.NESTEROV,
            )
    elif cfg.OPTIM.METHOD == "adam":
        optim = torch.optim.Adam(
            model.parameters(),
            lr=cfg.SOLVER.BASE_LR, 
            betas=(0.9, 0.999),
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    #scheduler = ReduceLROnPlateau(optim, 'min', patience=1,verbose=True)
    train_meter = utils.Train_meter(cfg)
    test_meter =utils.Test_meter(cfg) 
    return (
        model,
        optim,
        train_meter,
        test_meter,
    )
        

@torch.no_grad()   
def val_epoch(cfg, model, test_loader, test_meter, cur_epoch):
    model.eval()
    test_meter.time_start()
    for cur_iter, (inputs, labels) in enumerate(test_loader):
        test_meter.time_pause()
        test_meter.update_data()
        test_meter.time_start()
        
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)
        
        labels = labels.cuda()
        outputs = model(inputs)
      
        if cfg.MODEL.NUM_LABELS == 2:
            acc = utils.get_binary_acc(outputs, labels) 
        else:
            acc = utils.get_accuracy(outputs, labels) 
        
        acc = acc.item()
        batch_size = inputs.size(0)
        test_meter.update_states(acc, batch_size)
        
        test_meter.time_pause()
        test_meter.update_batch()
        test_meter.time_start()
    
    test_meter.update_epoch(cur_epoch)
    test_meter.reset()


def train_epoch(cfg, model, optim, train_loader, train_meter,cur_epoch):
    data_size = len(train_loader)
    train_meter.time_start()
    for cur_iter, (inputs, labels) in enumerate(train_loader):
        train_meter.time_pause()
        train_meter.update_data()
        train_meter.time_start()
        
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)
            
        lr = utils.get_lr_at_epoch(cfg, cur_epoch+float(cur_iter)/data_size)
        utils.set_lr(optim, lr)
        labels = labels.cuda()
        outputs = model(inputs)
      
        loss_func = nn.BCEWithLogitsLoss.cuda()
        loss = loss_func(outputs, labels)
        
        if cfg.MODEL.NUM_LABELS == 2:
            acc = utils.get_binary_acc(outputs, labels) 
        else:
            acc = utils.get_accuracy(outputs, labels) 
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        loss, acc = (
            loss.item(),
            acc.item(),
        )
        batch_size = inputs.size(0)
        train_meter.update_states(loss, acc, batch_size, lr)
        
        train_meter.time_pause()
        train_meter.update_batch()
        train_meter.time_start()
    
    train_meter.update_epoch(cur_epoch, cfg)
    train_meter.reset()

def train_net(cfg):
    utils.setup_logging(cfg.OUT_DIR)
    (
        model,
        optim,
        train_meter,
        test_meter,
    ) = build_model(cfg)
    data_container = Data_loader(cfg)
    logger.info("start load dataset")
    train_loader = data_container.construct_loader(
        cfg.RUN.TRAIN_BATCH_SIZE, cfg.RUN.NUM_WORKS, "train"
    )
    test_loader = data_container.construct_loader(
        cfg.RUN.TEST_BATCH_SIZE, cfg.RUN.NUM_WORKS, "test"
    )
    start_epoch = utils.load_train_checkpoint(cfg, model, optim)
    cudnn.benchmark = True
    
    logger.info("start epoch {}".format(start_epoch+1))

    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        train_epoch(cfg, model, optim, train_loader, train_meter, epoch)
        utils.save_checkpoint(cfg.OUT_DIR,cfg.CHECKPOINTS_FOLD, model, optim, epoch, cfg.NUM_GPUS)
        if cfg.SOLVER.ENABLE_VAL:
            val_epoch(cfg, model, test_loader, test_meter, epoch)
         
        
if __name__ == "__main__":
    train_net(cfg)