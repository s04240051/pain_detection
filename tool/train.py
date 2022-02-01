from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn 
import torch.backends.cudnn as cudnn

#import _init_path
import lib.model.two_stream as model_set
from lib.data_build import Data_loader
from lib.config_file import cfg
import lib.model.utils as utils

logger = utils.get_logger(__name__)
def build_model(cfg):
    
    model = eval(f"model_set.{cfg.MODEL.MODEL_NAME}(cfg).cuda()")
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
    
    train_meter = utils.Train_meter(cfg)
    test_meter =utils.Val_meter(cfg) 
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
    for cur_iter, (inputs, labels) in enumerate(tqdm(test_loader)):
        test_meter.time_pause()
        test_meter.update_data()
        test_meter.time_start()
        
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)
        
        labels = labels.cuda()
        labels = labels.unsqueeze(-1)
        outputs = model(inputs)

        loss_func = nn.BCEWithLogitsLoss().cuda()
        loss = loss_func(outputs, labels.float())

        if cfg.MODEL.NUM_LABELS == 2:
            acc = utils.get_binary_acc(outputs, labels) 
        else:
            acc = utils.get_accuracy(outputs, labels) 
        
        loss, acc = (
            loss.item(),
            acc.item(),
        )
        batch_size = inputs[0].size(0)
        test_meter.update_states(acc, loss, batch_size)
        
        test_meter.time_pause()
        test_meter.update_batch()
        test_meter.time_start()
    
    test_meter.update_epoch(cur_epoch)
    acc = test_meter.acc_meter.avg
    test_meter.reset()
    return acc

def train_epoch(cfg, model, optim, train_loader, train_meter,cur_epoch):
    model.train()
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
        labels = labels.unsqueeze(-1)
        outputs = model(inputs)
      
        loss_func = nn.BCEWithLogitsLoss().cuda()
        loss = loss_func(outputs, labels.float())
        
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
        batch_size = inputs[0].size(0)
        train_meter.update_states(loss, acc, batch_size, lr)
        
        train_meter.time_pause()
        train_meter.update_batch()
        train_meter.time_start()
    
    train_meter.update_epoch(cur_epoch, cfg)
    train_meter.reset()

def train_net(cfg):
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    utils.setup_logging(cfg.OUT_DIR)
    (
        model,
        optim,
        train_meter,
        test_meter,
    ) = build_model(cfg)
    data_container = Data_loader(cfg)
    logger.info("start load dataset")
    train_loader = data_container.construct_loader("train")
    test_loader = data_container.construct_loader("test")
    start_epoch = utils.load_train_checkpoint(cfg, model, optim)
    cudnn.benchmark = True
    
    logger.info("start epoch {}".format(start_epoch+1))
    best_pred = 0
    isbest = False
    for epoch in range(start_epoch+1, cfg.SOLVER.MAX_EPOCH):
        train_epoch(cfg, model, optim, train_loader, train_meter, epoch)
        if cfg.SOLVER.ENABLE_VAL:
            acc = val_epoch(cfg, model, test_loader, test_meter, epoch)
            isbest = acc > best_pred
            if isbest:
                best_pred = acc
                best_epoch = epoch 
                
        trigger_save = utils.save_policy(epoch, isbest, cfg)
        if trigger_save:
            utils.save_checkpoint(cfg.OUT_DIR,cfg.CHECKPOINTS_FOLD, model, optim, epoch, cfg.NUM_GPUS)   
    if cfg.SOLVER.ENABLE_VAL:
        logger.info("best model in {} epoch with acc {:.3f}".format(best_epoch, best_pred))
     
if __name__ == "__main__":
    train_net(cfg)
        