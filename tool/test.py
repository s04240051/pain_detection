import numpy as np
from tqdm import tqdm
import os
import torch
import torch.nn as nn 
import torch.backends.cudnn as cudnn


from lib.model.two_stream import two_stream_model
from lib.data_build import Data_loader
from lib.config_file import cfg
import lib.model.utils as utils

logger = utils.get_logger(__name__)
def build_model(cfg):
    if cfg.MODEL.MODEL_NAME == "two_lstm":
        model = two_stream_model(cfg).cuda()
    
    test_meter =utils.Test_meter(cfg) 
    return (
        model,
        test_meter,
    )
      

def test_net(cfg):
    utils.setup_logging(cfg.OUT_DIR)
    (
        model,
        test_meter,
    ) = build_model(cfg)
      
    data_container = Data_loader(cfg)
    logger.info("start load dataset")
    
    test_loader = data_container.construct_loader(
        cfg.RUN.TEST_BATCH_SIZE, cfg.RUN.NUM_WORKS, "test"
    )
    utils.load_test_checkpoint(cfg, model)
    cudnn.benchmark = True
    
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    
    val_epoch(cfg, model, test_loader, test_meter)
  

@torch.no_grad()   
def val_epoch(cfg, model, test_loader, test_meter):
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
    
    test_meter.update_epoch(-1, cfg)
    test_meter.reset()