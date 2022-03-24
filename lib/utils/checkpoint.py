import os
import torch
import torch.nn as nn

def load_train_checkpoint(cfg, model, optim):
    
    file_list = get_checkpoints_set(cfg)

    if cfg.TRAIN_INITIAL_WEIGHT != "":
        file_path = cfg.TRAIN_INITIAL_WEIGHT
        assert os.path.isfile(
            cfg.TRAIN_INITIAL_WEIGHT
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
            "Unknown way of loading checkpoint."\
            "Using with random initialization, only for debugging."
        )
        return 
    checkpoint = torch.load(file_path)
    test_epoch  = checkpoint["epoch"]
    model_dict = model.state_dict()
    new_dict = {k:v for k,v in checkpoint["model_state"].items() if k in model_dict.keys()}
    model_dict.update(new_dict)
    model.load_state_dict(model_dict)
    return test_epoch

def load_ft(file, model, fine_tune_layer=[]):
    assert os.path.isfile(file), "no checkpoints file found"
    pre_dict = torch.load(file)
    model_dict = model.state_dict()
    new_dict = {k: v for k, v in pre_dict.items() if k in model_dict.keys()}
    model_dict.update(new_dict)
    model.load_state_dict(model_dict)
    
    paraname_set = []
    for prefix in fine_tune_layer:
        for name, para in model.named_parameters():
            if name.startswith(prefix):
                print('  Finetuning parameter: {}'.format(name))
                paraname_set.append(name)
    for name, para in model.named_parameters():
        if name not in paraname_set:
            para.requires_grad = False
    if not fine_tune_layer:
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm3d):
                module.training = False

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

def save_policy(cur_epoch, isbest, cfg):
    if isbest:
        return True
    save_step = cfg.RUN.SAVE_STEP
    if cur_epoch >= cfg.SOLVER.MAX_EPOCH-cfg.RUN.SAVE_LAST:
        return True
    elif ((cur_epoch) % save_step) == 0:
        return True
    else:
        return False
class best_policy:
    def __init__(self, cfg):
        if (cfg.SOLVER.LR_POLICY).startswith("steps"):
            self.mark_start = cfg.SOLVER.STEPS[1]
        else:
            self.mark_start = cfg.WARMUP_EPOCHS
        self.best_pred = 0
        self.isbest = False
        self.best_epoch = 0
    def update(self, cur_epoch, acc):
        if cur_epoch >= self.mark_start and acc > self.best_pred:
            self.best_pred = acc
            self.best_epoch = cur_epoch
            return True
        else:
            return False
        
