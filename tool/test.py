import numpy as np
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from lib.data_build.data_utils import loss_weight

import lib.model.two_stream as model_set
from lib.data_build import Data_loader
#import lib.model.utils as utils
import lib.utils.logging as log
import lib.utils.checkpoint as cu
from lib.utils.meter import Test_meter
import lib.utils.solver as sol

logger = log.get_logger(__name__)


def build_model(cfg):
    model = eval(f"model_set.{cfg.MODEL.MODEL_NAME}(cfg).cuda()")

    test_meter = Test_meter(cfg)
    return (
        model,
        test_meter,
    )


def test_net(cfg):
    log.setup_logging(cfg.OUT_DIR)
    (
        model,
        test_meter,
    ) = build_model(cfg)

    data_container = Data_loader(cfg)
    logger.info("start load dataset")

    test_loader, loss_weight = data_container.construct_loader("test")
    #labels = (np.array(test_loader.dataset.label_list)[:, -1]).astype(int)
    test_epoch = cu.load_test_checkpoint(cfg, model)
    cudnn.benchmark = True

    logger.info("Testing model for {} iterations".format(len(test_loader)))

    val_epoch(cfg, model, test_loader, test_meter, test_epoch)


@torch.no_grad()
def val_epoch(cfg, model, test_loader, test_meter, epoch=-1):
    model.eval()
    test_meter.time_start()
    for cur_iter, (inputs, labels, start_name) in enumerate(tqdm(test_loader)):
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
        if cfg.MODEL.MODEL_NAME == "Two_stream_fusion" and cfg.MODEL.FUSION_METHOD == "late":
            loss_func = nn.BCELoss().cuda()
        else:
            loss_func = nn.BCEWithLogitsLoss().cuda()
        loss = loss_func(outputs, labels.float())

        if cfg.MODEL.NUM_LABELS == 2:
            acc = sol.get_binary_acc(outputs, labels)
        else:
            acc = sol.get_accuracy(outputs, labels)

        loss, acc = (
            loss.item(),
            acc.item(),
        )
        batch_size = inputs[0].size(0)
        outputs = torch.sigmoid(outputs)
        outputs = outputs.cpu()
        labels  = labels.cpu()
        
        test_meter.update_states(loss, acc, batch_size, outputs, labels, start_name)

        test_meter.time_pause()
        test_meter.update_batch()
        test_meter.time_start()

    test_meter.update_epoch(epoch)
    acc = test_meter.acc_meter.avg
    f1_score = test_meter.f1
    test_meter.reset()
    return acc, f1_score


def test_full(cfg):
    log.setup_logging(cfg.OUT_DIR)
    checkpoints_set = cu.get_checkpoints_set(cfg)
    assert len(checkpoints_set) > 0, f"no checkpoints file avalible in {cfg.CHECKPOINTS_FOLD}"
    
    (
        model,
        test_meter,
    ) = build_model(cfg)

    data_container = Data_loader(cfg)
    logger.info("start load dataset")

    test_loader = data_container.construct_loader("test")
    cudnn.benchmark = True
  
    best_preds = 0
    best_f1 = 0
    for file in checkpoints_set:
        checkpoint = torch.load(file)
        epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint['model_state'])
      
        acc, f1_score = val_epoch(cfg, model, test_loader, test_meter, epoch)
        if acc > best_preds:
            best_preds, best_f1, best_epoch = acc, f1_score, epoch
    logger.info("best model in {} epoch with acc {:.3f}, f1 score {:.3f}".format(
        best_epoch, best_preds, best_f1))
