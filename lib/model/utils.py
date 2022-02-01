from sklearn.metrics import f1_score
import os
import time
import math
import datetime
import pandas as pd
import torch
import logging


def load_train_checkpoint(cfg, model, optim):
    
    file_list = get_checkpoints_set(cfg)

    if cfg.TRAIN_INITIAL_WEIGHT != "":
        file_path = cfg.TRAIN_INITIAL_WEIGHT
        assert os.path.isfile(
            cfg.TEST_INITIAL_WEIGHT
        ), "Checkpoint '{}' not found".format(file_path) 

    elif file_list:
        file_path = sorted(file_list)[-1]

    else:
        print("No checkpoint file found.")
        return -1
    checkpoint = torch.load(file_path)
    start_epoch = checkpoint['epoch']
    
    model.load_state_dict(checkpoint['model_state'])
    optim.load_state_dict(checkpoint['optimizer_state'])
    return start_epoch

def load_test_checkpoint(cfg, model):
    
    if cfg.TEST_INITIAL_WEIGHT != "":
        file_path = cfg.TEST_INITIAL_WEIGHT
        assert os.path.isfile(
            cfg.TEST_INITIAL_WEIGHT
        ), "Checkpoint '{}' not found".format(file_path) 
    elif get_checkpoints_set(cfg):
        file_path =  sorted(get_checkpoints_set(cfg))[-1]
    elif cfg.TRAIN_INITIAL_WEIGHT != "":
        file_path = cfg.TRAIN_INITIAL_WEIGHT
        assert os.path.isfile(
            cfg.TEST_INITIAL_WEIGHT
        ), "Checkpoint '{}' not found".format(file_path) 
    else:
        print(
            "Unknown way of loading checkpoint. Using with random initialization, only for debugging."
        )
        return 
    checkpoint = torch.load(file_path)
    test_epoch  = checkpoint["epoch"]
    model.load_state_dict(checkpoint['model_state'])
    return test_epoch
    
def save_checkpoint(out_dir, checkpoints_fold, model, optim, cur_epoch, num_gpus):
    sd = model.module.state_dict() if num_gpus > 1 else model.state_dict()
    checkpoint = {
        "epoch": cur_epoch,
        "model_state": sd,
        "optimizer_state": optim.state_dict(),

    }
    name = "checkpoint_epoch_{:05d}.pt".format(cur_epoch)
    path_to_checkpoints = os.path.join(
        get_checkpoints_path(out_dir, checkpoints_fold), name)
    torch.save(checkpoint, path_to_checkpoints)


def get_checkpoints_path(out_dir, sub_name):
    assert os.path.isdir(out_dir)
    path = os.path.join(out_dir, sub_name)
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def get_checkpoints_set(cfg):
    checkpoint_dir = get_checkpoints_path(cfg.OUT_DIR, cfg.CHECKPOINTS_FOLD)
    if os.path.isdir(checkpoint_dir):
        checkpoint_list = [os.path.join(checkpoint_dir,file) for file in os.listdir(
            checkpoint_dir) if file.endswith(".pt")]
        return checkpoint_list
    else:
        return []


def get_accuracy(outputs, label, topk=1):
    _, indies = torch.topk(outputs, k=topk, sorted=True)
    label = torch.reshape(label, [-1, 1])
    correct = (indies == label).any(axis=1)*1.
    acc = torch.mean(correct)
    return acc


def get_binary_acc(outputs, labels):
    labels = torch.reshape(labels, [-1, 1])
    outputs = (outputs > 0)
    acc = torch.mean((outputs == labels)*1.)
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
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Train_meter:
    def __init__(self, cfg):
        self.data_meter = AverageMeter()
        self.batch_meter = AverageMeter()
        self.loss_meter = AverageMeter()
        self.acc_meter = AverageMeter()
        self.lr = None
        self.max_epoch = cfg.SOLVER.MAX_EPOCH
        self.logger = get_logger(__name__)
        self.record_path = self.init_record("train", cfg)
        
    
    def init_record(self, split, cfg):
        record_dir = os.path.join(cfg.OUT_DIR, split+"_record")
        record_name = eval(f"cfg.{split.upper()}_RECORD")
        
        record_path = os.path.join(record_dir, record_name)
        
        if record_name == "" or not os.path.isfile(record_path):
            record_name = "{}_record{:03}.csv".format(split, len(os.listdir(record_dir)))
            record_path = os.path.join(record_dir, record_name)
        self.logger.info("save {} record in {}".format(split, record_path))
        return record_path
        
    
    def time_start(self):
        self.start = time.perf_counter()
        self._pause = None

    def time_pause(self):
        if self._pause is not None:
            raise ValueError("Trying to pause a Timer that is already paused!")
        self.pause = time.perf_counter()

    def update_data(self):
        if self._pause is not None:
            end_time = self._pause
        else:
            end_time = time.perf_counter()
        self.data_meter.update(end_time-self.start)

    def update_batch(self):
        if self._pause is not None:
            end_time = self._pause
        else:
            end_time = time.perf_counter()
        self.batch_meter.update(end_time-self.start)

    def update_states(self, loss, acc, batch_size, lr):
        self.loss_meter.update(loss, batch_size)
        self.acc_meter.update(acc, batch_size)
        self.lr = lr

    def update_epoch(self, cur_epoch, cfg):
        eta_sec = (self.batch_meter.sum + self.data_meter.sum) * \
            (self.max_epoch-cur_epoch)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "train_epoch",
            "epoch": "{}/{}".format(cur_epoch, cfg.SOLVER.MAX_EPOCH),
            "dt_data": round(self.data_meter.avg, 2),
            "dt_net": round(self.batch_meter.avg, 2),
            "lr": self.lr,
            "eta": eta,
            "loss": round(self.loss_meter.avg, 3),
            "accuracy": round(self.acc_meter.avg, 3)
        }
        
        self.record_info(stats, self.record_path)
    
    def record_info(self, info, filename):
        result = "|".join([f"{key} {item}" for key, item in info.items()])

        
        self.logger.info("json states: {:s}".format(result))

        df = pd.DataFrame([info])

        if not os.path.isfile(filename):
            df.to_csv(filename, index=False)
        else:  # else it exists so append without writing the header
            df.to_csv(filename, mode='a', header=False, index=False)
    
    def reset(self):
        self.acc_meter.reset()
        self.batch_meter.reset()
        self.data_meter.reset()
        self.loss_meter.reset()
        self.lr = None

class test_meter(Train_meter):
    def __init__(self, cfg):
        
        self.data_meter = AverageMeter()
        self.batch_meter = AverageMeter()
        self.loss_meter = AverageMeter()
        self.acc_meter = AverageMeter()
        self.logger = get_logger(__name__)
        self.record_path = self.init_record("test", cfg)
        self.preds = torch.tensor([])
    
    def update_states(self, loss, acc, batch_size, preds):
        self.acc_meter.update(acc, batch_size)
        self.loss_meter.update(loss, batch_size)
        self.preds = torch.concat((self.preds, preds))
    
    def update_epoch(self, cur_epoch, cfg, labels):
        self.f1 = f1_score(labels, self.preds>0, average="weighted")
        stats = {
            "_type": "test_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, cfg.SOLVER.MAX_EPOCH),
            "dt_data": round(self.data_meter.avg, 2),
            "dt_net": round(self.batch_meter.avg, 2),
            "accuracy": round(self.acc_meter.avg, 3),
            "loss": round(self.loss_meter.avg, 3),
            "f1_score": round(self.f1, 3),
        }
        
        self.record_info(stats, self.record_path)
    
    def reset(self):
        self.acc_meter.reset()
        self.batch_meter.reset()
        self.data_meter.reset()
        self.loss_meter.reset()
        self.preds = torch.tensor([])
        self.f1 = 0 
class Val_meter(Train_meter):
    def __init__(self, cfg):
        #super(Val_meter, self).__init__(cfg)
        self.data_meter = AverageMeter()
        self.batch_meter = AverageMeter()
        self.loss_meter = AverageMeter()
        self.acc_meter = AverageMeter()
        self.logger = get_logger(__name__)
        self.record_path = self.init_record("val", cfg)
    
    def update_states(self, acc,loss, batch_size):
        self.acc_meter.update(acc, batch_size)
        self.loss_meter.update(loss, batch_size)
    def update_epoch(self, cur_epoch, cfg):
        stats = {
            "_type": "test_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, cfg.SOLVER.MAX_EPOCH),
            "dt_data": round(self.data_meter.avg, 2),
            "dt_net": round(self.batch_meter.avg, 2),
            "accuracy": round(self.acc_meter.avg, 3),
            "loss": round(self.loss_meter.avg, 3),
        }
        
        self.record_info(stats, self.record_path)

    def reset(self):
        self.acc_meter.reset()
        self.batch_meter.reset()
        self.data_meter.reset()
        self.loss_meter.reset()

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


def get_logger(name):

    return logging.getLogger(name)


def setup_logging(log_path):

    # 获取logger对象,取名
    logger = logging.getLogger()
    # 输出DEBUG及以上级别的信息，针对所有输出的第一层过滤
    logger.setLevel(level=logging.DEBUG)
    # 获取文件日志句柄并设置日志级别，第二层过滤
    log_path = get_checkpoints_path(log_path, "logger")
    log_path = os.path.join(log_path, "stdout.log")
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    # 生成并设置文件日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # console相当于控制台输出，handler文件输出。获取流句柄并设置日志级别，第二层过滤
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    # 为logger对象添加句柄
    logger.addHandler(handler)
    logger.addHandler(console)
