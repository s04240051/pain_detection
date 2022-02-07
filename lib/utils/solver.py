import torch
import torch.nn as nn
import math

def get_binary_acc(outputs, labels):
    labels = torch.reshape(labels, [-1, 1])
    outputs = (outputs > 0)
    acc = torch.mean((outputs == labels)*1.)
    return acc

def get_accuracy(outputs, label, topk=1):
    _, indies = torch.topk(outputs, k=topk, sorted=True)
    label = torch.reshape(label, [-1, 1])
    correct = (indies == label).any(axis=1)*1.
    acc = torch.mean(correct)
    return acc

def save_policy(cur_epoch, isbest, cfg):
    if isbest and cur_epoch > cfg.SOLVER.WARMUP_EPOCHS:
        return True
    save_step = cfg.RUN.SAVE_STEP
    if cur_epoch >= cfg.SOLVER.MAX_EPOCH-cfg.RUN.SAVE_LAST:
        return True
    elif ((cur_epoch) % save_step) == 0:
        return True
    else:
        return False

def get_lr_at_epoch(cfg, cur_epoch):
    """
    Retrieve the learning rate of the current epoch with the option to perform
    warm up in the beginning of the training stage.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.
    """
    lr = get_lr_func(cfg.SOLVER.LR_POLICY)(cfg, cur_epoch)
    # Perform warm up.
    if cur_epoch < cfg.SOLVER.WARMUP_EPOCHS:
        lr_start = cfg.SOLVER.WARMUP_START_LR
        lr_end = get_lr_func(cfg.SOLVER.LR_POLICY)(
            cfg, cfg.SOLVER.WARMUP_EPOCHS
        )
        alpha = (lr_end - lr_start) / cfg.SOLVER.WARMUP_EPOCHS
        lr = cur_epoch * alpha + lr_start
    return lr


def lr_func_cosine(cfg, cur_epoch):
    """
    Retrieve the learning rate to specified values at specified epoch with the
    cosine learning rate schedule. Details can be found in:
    Ilya Loshchilov, and  Frank Hutter
    SGDR: Stochastic Gradient Descent With Warm Restarts.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.
    """
    offset = cfg.SOLVER.WARMUP_EPOCHS if cfg.SOLVER.COSINE_AFTER_WARMUP else 0.0
    assert cfg.SOLVER.COSINE_END_LR < cfg.SOLVER.BASE_LR
    return (
        cfg.SOLVER.COSINE_END_LR
        + (cfg.SOLVER.BASE_LR - cfg.SOLVER.COSINE_END_LR)
        * (
            math.cos(
                math.pi * (cur_epoch - offset) /
                (cfg.SOLVER.MAX_EPOCH - offset)
            )
            + 1.0
        )
        * 0.5
    )

def lr_func_steps_with_relative_lrs(cfg, cur_epoch):
    """
    Retrieve the learning rate to specified values at specified epoch with the
    steps with relative learning rate schedule.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.
    """
    ind = get_step_index(cfg, cur_epoch)
    return cfg.SOLVER.LRS[ind] * cfg.SOLVER.BASE_LR


def get_step_index(cfg, cur_epoch):
    """
    Retrieves the lr step index for the given epoch.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.
    """
    steps = cfg.SOLVER.STEPS + [cfg.SOLVER.MAX_EPOCH]
    for ind, step in enumerate(steps):  # NoQA
        if cur_epoch < step:
            break
    return ind - 1

def get_lr_func(lr_policy):
    """
    Given the configs, retrieve the specified lr policy function.
    Args:
        lr_policy (string): the learning rate policy to use for the job.
    """
    policy = "lr_func_" + lr_policy
    if policy not in globals():
        raise NotImplementedError("Unknown LR policy: {}".format(lr_policy))
    else:
        return globals()[policy]

def set_lr(optimizer, new_lr):
    """
    Sets the optimizer lr to the specified value.
    Args:
        optimizer (optim): the optimizer using to optimize the current network.
        new_lr (float): the new learning rate to set.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params :
        num: int，the number of loss
        x: multi-task loss
    
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

def loss_builder(loss_w, data_type):
    if data_type == "aux":
        weight1 = None if loss_w[0] is None else loss_w[0][1]
        loss_func1 = nn.BCEWithLogitsLoss(pos_weight=weight1).cuda()
        loss_func2 = nn.CrossEntropyLoss(weight=loss_w[1]).cuda()
        return [loss_func1, loss_func2]
    else:
        if data_type == "simple":
            weight1 = None if loss_w[0] is None else loss_w[0][1]
            loss_func = nn.BCEWithLogitsLoss(pos_weight=weight1).cuda()
            return [loss_func]
        else:
            loss_func = nn.CrossEntropyLoss(weight=loss_w[1]).cuda()
            return [loss_func]