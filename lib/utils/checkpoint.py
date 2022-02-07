import os
import torch

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