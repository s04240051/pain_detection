import torch
import os
import time
import datetime
import pandas as pd
from sklearn.metrics import f1_score
from collections import defaultdict
from .logging import get_logger

logger = get_logger(__name__)
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
        self.cfg = cfg
        self.info = defaultdict(lambda:AverageMeter())
        self.lr = None
        self.max_epoch = cfg.SOLVER.MAX_EPOCH
        self.record_path = self.init_record("train")
        
    def init_record(self, split):
        record_dir = os.path.join(self.cfg.OUT_DIR, split+"_record")
        record_name = eval(f"self.cfg.{split.upper()}_RECORD")
        
        record_path = os.path.join(record_dir, record_name)
        
        if record_name == "" or not os.path.isfile(record_path):
            record_name = self.cfg.CHECKPOINTS_FOLD + ".csv"
            #record_name = "{}_record{:03}.csv".format(split, len(os.listdir(record_dir)))
            record_path = os.path.join(record_dir, record_name)
        logger.info("save {} record in {}".format(split, record_path))
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

    def update_states(self, batch_size, lr, **param):
        
        for key, value in param.items():
            self.info[key].update(value, batch_size)
        self.lr = lr

    def update_epoch(self, cur_epoch):
        eta_sec = (self.batch_meter.sum + self.data_meter.sum) * \
            (self.max_epoch-cur_epoch)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        epoch_sec = round(self.data_meter.sum+self.batch_meter.sum, 2)
        epoch_time = str(datetime.timedelta(seconds=int(epoch_sec)))
        states = {
            "_type": "train_epoch",
            "epoch": "{}/{}".format(cur_epoch, self.max_epoch),
            "dt_data": round(self.data_meter.avg, 2),
            "dt_net": round(self.batch_meter.avg, 2),
            "epoch_time": epoch_time,
            "lr": self.lr,
            "eta": eta,
        }
            
        state2 = {key: round(value.avg, 3) for key, value in self.info.items() }
        final_state = {**states, **state2}
        self.record_info(final_state, self.record_path)
    
    def record_info(self, info, filename):
        result = "|".join([f"{key} {item}" for key, item in info.items()])

        
        logger.info("json states: {:s}".format(result))

        df = pd.DataFrame([info])

        if not os.path.isfile(filename):
            df.to_csv(filename, index=False)
        else:  # else it exists so append without writing the header
            df.to_csv(filename, mode='a', header=False, index=False)
    
    def reset(self):
        
        self.batch_meter.reset()
        self.data_meter.reset()
        self.info.clear()
        self.lr = None

class test_meter(Train_meter):
    def __init__(self, cfg):
        
        self.data_meter = AverageMeter()
        self.batch_meter = AverageMeter()
        self.loss_meter = AverageMeter()
        self.acc_meter = AverageMeter()
        self.cfg = cfg
        self.record_path = self.init_record("test")
        self.preds = torch.tensor([])
    
    def update_states(self, loss, acc, batch_size, preds):
        self.acc_meter.update(acc, batch_size)
        self.loss_meter.update(loss, batch_size)
        self.preds = torch.concat((self.preds, preds))
    
    def update_epoch(self, cur_epoch, cfg, labels):
        self.f1 = f1_score(labels, self.preds>0, average="weighted")
        stats = {
            "_type": "test_epoch",
            "epoch": "{}/{}".format(cur_epoch, cfg.SOLVER.MAX_EPOCH),
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
        
        self.data_meter = AverageMeter()
        self.batch_meter = AverageMeter()
        self.info = defaultdict(lambda: AverageMeter())
        self.cfg = cfg
        self.record_path = self.init_record("val")
    
    def update_states(self, batch_size, **param):
        for key, value in param.items():
            self.info[key].update(value, batch_size)
        
    def update_epoch(self, cur_epoch):
        epoch_sec = round(self.data_meter.sum+self.batch_meter.sum, 2)
        epoch_time = str(datetime.timedelta(seconds=int(epoch_sec)))
        states = {
            "_type": "test_epoch",
            "epoch": "{}/{}".format(cur_epoch, self.cfg.SOLVER.MAX_EPOCH),
            "dt_data": round(self.data_meter.avg, 2),
            "dt_net": round(self.batch_meter.avg, 2),
            "epoch_time": epoch_time,
        }
        states1 = {key: round(value.avg, 3) for key, value in self.info.items()}
        final_state = {**states, **states1}
        self.record_info(final_state, self.record_path)

    def reset(self):
        
        self.batch_meter.reset()
        self.data_meter.reset()
        self.info.clear()
