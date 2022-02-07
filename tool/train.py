from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn 
import torch.backends.cudnn as cudnn
import lib.model.two_stream as model_set
from lib.data_build import Data_loader
from lib.config_file import cfg
import lib.utils.logging as log
import lib.utils.checkpoint as cu
from lib.utils.meter import Train_meter, Val_meter
import lib.utils.solver as sol
#import lib.model.utils as utils

logger = log.get_logger(__name__)
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
    
    train_meter = Train_meter(cfg)
    test_meter = Val_meter(cfg) 
    return (
        model,
        optim,
        train_meter,
        test_meter,
    )
        

@torch.no_grad()   
def val_epoch(
    cfg, 
    model, 
    test_loader, 
    test_meter, 
    cur_epoch,
    data_type,
    loss_pack, 
    loss_dict=None
    ):

    model.eval()
    if loss_dict is not None:
        awl = sol.AutomaticWeightedLoss(2).cuda()
        awl.load_state_dict(loss_dict)
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
        
        if isinstance(labels, list):
            labels[0] = (labels[0].unsqueeze(-1)).cuda()
            labels[1] = labels[1].cuda()
        else:
            labels = labels.cuda()
            if data_type == "simple":
                labels = labels.unsqueeze(-1) 
        outputs = model(inputs)

        if isinstance(outputs, list):
            out1, out2 = outputs[0], outputs[1]
            loss_func1 = loss_pack[0]
            loss_func2 = loss_pack[1]
            
            acc1 = sol.get_binary_acc(out1, labels[0])
            acc2 = sol.get_accuracy(out2, labels[1])
            loss1 = loss_func1(out1, labels[0].float())
            loss2 = loss_func2(out2, labels[1])
            loss = awl(loss1, loss2)
        else:
            if data_type == "simple":
                loss_func = loss_pack[0]
                loss = loss_func(outputs, labels.float())
                acc = sol.get_binary_acc(outputs, labels)
            else:
                loss_func = loss_pack[1]
                loss = loss_func(outputs, labels)
                acc = sol.get_accuracy(outputs, labels)

     
        batch_size = inputs[0].size(0)
        if data_type in ["simple", "diff"]:
            loss, acc = (
                loss.item(),
                acc.item(),
            )
            test_meter.update_states(batch_size, loss=loss, acc=acc)
        else:
            test_meter.update_states(
                batch_size,
                
                loss = loss.item(),
                loss1 = loss1.item(),
                loss2 = loss2.item(),
                acc1 = acc1.item(),
                acc2 = acc2.item(),
            )
        
        test_meter.time_pause()
        test_meter.update_batch()
        test_meter.time_start()
    
    test_meter.update_epoch(cur_epoch)
    if data_type in ["simple", "diff"]:
        accuracy_score = test_meter.info["acc"].avg
    else:
        accuracy_score =test_meter.info["acc1"].avg 
    test_meter.reset()
    return accuracy_score

def train_epoch(
    cfg, 
    model, 
    optim, 
    train_loader, 
    train_meter,
    cur_epoch, 
    data_type,
    loss_pack,
    ):
    model.train()
    data_size = len(train_loader)
    train_meter.time_start()
    for cur_iter, (inputs, labels) in enumerate(tqdm(train_loader)):
        train_meter.time_pause()
        train_meter.update_data()
        train_meter.time_start()
        
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)
            
        lr = sol.get_lr_at_epoch(cfg, cur_epoch+float(cur_iter)/data_size)
        sol.set_lr(optim, lr)
        if isinstance(labels, list):
            labels[0] = (labels[0].unsqueeze(-1)).cuda()
            labels[1] = labels[1].cuda()
        else:
            labels = labels.cuda()
            if data_type == "simple":
                labels = labels.unsqueeze(-1) 

        outputs = model(inputs)
        if isinstance(outputs, list):
            out1, out2 = outputs[0], outputs[1]
            loss_func1 = loss_pack[0]
            loss_func2 = loss_pack[1]
            awl = sol.AutomaticWeightedLoss(2).cuda()

            acc1 = sol.get_binary_acc(out1, labels[0])
            acc2 = sol.get_accuracy(out2, labels[1])
            loss1 = loss_func1(out1, labels[0].float())
            loss2 = loss_func2(out2, labels[1])
            loss = awl(loss1, loss2)
        else:
            if data_type == "simple":
                loss_func = loss_pack[0]
                loss = loss_func(outputs, labels.float())
                acc = sol.get_binary_acc(outputs, labels)
            else:
                loss_func = loss_pack[1]
                loss = loss_func(outputs, labels)
                acc = sol.get_accuracy(outputs, labels)
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        batch_size = inputs[0].size(0)
        if data_type in ["simple", "diff"]:
            loss, acc = (
                loss.item(),
                acc.item(),
            )
            train_meter.update_states(batch_size, lr, loss=loss, acc=acc)
        else:
            
            train_meter.update_states(
                batch_size,
                lr,
                loss = loss.item(),
                loss1 = loss1.item(),
                loss2 = loss2.item(),
                acc1 = acc1.item(),
                acc2 = acc2.item(),
            )
        
        train_meter.time_pause()
        train_meter.update_batch()
        train_meter.time_start()
    
    train_meter.update_epoch(cur_epoch)
    train_meter.reset()
    return awl.state_dict() if data_type == "aux" else None

def train_net(cfg):
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    torch.cuda.manual_seed(cfg.RNG_SEED)

    log.setup_logging(cfg.OUT_DIR)
    (
        model,
        optim,
        train_meter,
        test_meter,
    ) = build_model(cfg)
    data_container = Data_loader(cfg)
    logger.info("start load dataset")
    train_loader, loss_wtrain = data_container.construct_loader("train")
    test_loader, loss_wtest = data_container.construct_loader("test")
    start_epoch = cu.load_train_checkpoint(cfg, model, optim)
    cudnn.benchmark = True
    
    logger.info("start epoch {}".format(start_epoch+1))
    best_pred = 0
    isbest = False
    data_type = cfg.DATA.DATA_TYPE
    loss_pack_train = sol.loss_builder(loss_wtrain, data_type)
    loss_pack_test = sol.loss_builder(loss_wtest, data_type)
    
    for epoch in range(start_epoch+1, cfg.SOLVER.MAX_EPOCH):
        loss_dict = train_epoch(
            cfg, 
            model, 
            optim, 
            train_loader, 
            train_meter, 
            epoch, 
            data_type,
            loss_pack_train
        )
        if cfg.SOLVER.ENABLE_VAL:
            acc = val_epoch(
                cfg, 
                model, 
                test_loader, 
                test_meter, 
                epoch, 
                data_type,
                loss_pack_test,
                loss_dict=loss_dict,
            )
                
            isbest = acc > best_pred
            if isbest:
                best_pred = acc
                best_epoch = epoch 
                
        trigger_save = cu.save_policy(epoch, isbest, cfg)
        if trigger_save:
            cu.save_checkpoint(
                cfg.OUT_DIR,
                cfg.CHECKPOINTS_FOLD, 
                model, 
                optim, 
                epoch, 
                cfg.NUM_GPUS,
            )   
    if cfg.SOLVER.ENABLE_VAL:
        logger.info("best model in {} epoch with acc {:.3f}".format(best_epoch, best_pred))
     
if __name__ == "__main__":
    train_net(cfg)
        