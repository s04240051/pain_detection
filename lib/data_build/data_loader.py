
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as T_func
import torch.nn as nn
from PIL import Image
import torch
import numpy as np
import pandas as pd
from math import cos, sin
import os
from lib.model.utils import get_logger 
logger = get_logger(__name__)

class Data_loader:
    def __init__(self, cfg):
        if cfg.FORMAT_DATASET and cfg.DATASET_PATH != "":
            data_path = cfg.DATASET_PATH
            ann_path = os.path.join(data_path, "annotation")
            self.crop_img = os.path.join(data_path, "crop_img")
            self.data_file = os.path.join(ann_path, "kp_valid")
            self.label_file = os.path.join(ann_path, "split")
        else:
            self.data_file = cfg.KEYPOINT_FILE
            self.label_file = cfg.TRAIN_TEST_SPLIT
            self.crop_img = cfg.CROP_IMAGE
        self.model_type = cfg.MODEL.TYPE
        assert self.model_type in ["two_stream", "rgb", "kp"], f"model type not support"
        self.img_size = cfg.MODEL.IMG_SIZE
        self.clip_len = cfg.CLIP_LENGTH
        self.cfg = cfg
    
    def label_prepare(self, split):
        file = os.path.join(self.label_file, split+".h5")
        step = self.clip_len if split == "test" else None
        container = []
        label_info = {"pain":0, "not_pain":0}
        df = pd.read_hdf(file, "df_with_missing")
        df = df.applymap(lambda item: int(item))
        for i in range(len(df)):
            line = df.iloc[i]
            clip_set = self.clip_extract(line, step)
            self.label_update(label_info, len(clip_set), line["label"])
            container.extend(clip_set)
        label_info[split] = len(container)
        label_item = [f"{key}: {item}" for key, item in label_info.items()]
        label_item = ",".join(label_item) 
        logger.info(label_item)
        return container
    
    def label_update(self, label_info, num_clip, label):
        name_mp = ["not_pain", "pain"]
        key = name_mp[label]
        label_info[key] = label_info[key] + num_clip

    def clip_extract(self, line, step=None):
        if not step:
            step = 1 if line["length"] <= 2*self.clip_len else 2
        return [[line.name, i, i+self.clip_len, line["label"]] for i in range(line["starts"], line["ends"]-self.clip_len, step)]

    def construct_loader(self, split):
        if split == "train":
            shuffle = True
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        clip_df_set = self.label_prepare(split)
        dataset = Pain_dataset(
            clip_df_set, 
            self.data_file,
            self.crop_img, 
            self.img_size,
            self.model_type,
            self.cfg,
        )
        loader = DataLoader(
            dataset,
            self.cfg.RUN.TRAIN_BATCH_SIZE,
            shuffle,
            num_workers=self.cfg.RUN.NUM_WORKS,
            pin_memory=True,
            drop_last=drop_last,
            
        )
        return loader


class Pain_dataset(Dataset):
    def __init__(
        self,
        label_list,
        video_file,
        crop_img,
        img_size,
        model_type,
        cfg,
    ):

        self.label_list = label_list
        self.video_file = video_file
        self.crop_img = crop_img

        self.img_size = img_size
        self.data_dict = {}
        self.bbox_col = ["x", "y", "w", "h"]
        self.model_type = model_type
        self.cfg = cfg
    
    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        clip = self.label_list[index]
        file_name = clip[0]
        if file_name in self.data_dict:
            video_df = self.data_dict[file_name]

        else:
            video_df = pd.read_hdf(
                os.path.join(self.video_file, file_name), "df_with_missing"
            )
            self.data_dict[file_name] = video_df
        
        input_data = []
        video_name = os.path.join(self.crop_img, file_name.split(".")[0])
        clip_inf = video_df.iloc[clip[1]:clip[2]]
        aug_para = aug_param_builder(clip_inf, video_name) if self.cfg.DATA.AUG else []
        
        if self.model_type in ["two_stream", "rgb"]:
            frames = self.frames_builder(
                clip_inf[self.bbox_col], video_name, aug_para
            )
            input_data.append(frames)
            
        if self.model_type in ["two_stream", "kp"]:
            feature  = clip_inf.iloc[:, :-4].values
            kps = self.keypoints_builder(
                feature, aug_para
            )
            input_data.append(kps)
        label = clip[-1]
        
        return input_data, int(label)  
        
    def keypoints_builder(self, kp_inf, aug_para):
        if not aug_para :
            return torch.from_numpy(kp_inf.copy()).float()
        kp_inf = kp_horizontal_flip(kp_inf, aug_para[2], aug_para[3])
        kp_inf = kp_rotate(kp_inf, aug_para)
        kp_inf = kp_normal(kp_inf)
        return torch.from_numpy(kp_inf.copy()).float()
    
    def frames_builder(self, frame_inf, video_name, aug_para):
        crop_threshold = self.cfg.DATA.CROP_THRESHOLD
        frames = []
        mean = self.cfg.DATA.MEAN
        std = self.cfg.DATA.STD
        for img_index in frame_inf.index:
            line = frame_inf.loc[img_index]
            path = os.path.join(video_name, img_index)
            img = Image.open(path)
            img_w, img_h = img.size

            for crop_size in crop_threshold:
                if (line[["w", "h"]] < crop_size).all() and img_w != crop_size:
                    crop_x = np.clip(
                        line["x"] - (crop_size-line["w"]) /
                        2, 0, img_w - crop_size
                    )
                    crop_y = np.clip(
                        line["y"] - (crop_size-line["h"]) /
                        2, 0, img_h - crop_size
                    )
                    img = img.crop(
                        (crop_x, crop_y, crop_x+crop_size, crop_y+crop_size)
                    )
                    break
            img = img.resize(self.img_size, Image.ANTIALIAS)
            frames.append(img)
        
        frames = torch.as_tensor(np.stack(frames))
        frames = tensor_normalize(frames, mean, std)
        frames = frames.permute(0,3,1,2)

        if not aug_para:
            return frames
        else:
            return frame_augmentation(frames, aug_para)
        

def aug_param_builder(clip_inf, video_name):
    path = os.path.join(video_name, clip_inf.index[0])
    img = Image.open(path)
    img_w, img_h = img.size
    rotate_degree =  np.random.randint(-180, 180)
    rotate_prob, flip_prob = np.random.uniform(), np.random.uniform()
    aug_para = [rotate_degree, rotate_prob, flip_prob, img_w, img_h]
    return aug_para

def kp_normal(lines):
    x_index = np.arange(0, lines.shape[1], 2)
    y_index = np.arange(1, lines.shape[1], 2)
    spine_len = np.sqrt(
        (lines[:, 8] - lines[:, 6])**2 + 
        (lines[:, 9] - lines[:, 7])**2
    )

    root_x, root_y = (
        (lines[:, 6] + lines[:, 8]) / 2,
        (lines[:, 7] + lines[:, 9]) / 2,
    )
    
    lines[:, x_index] = np.apply_along_axis(lambda x: (x-root_x)/spine_len, 0, lines[:,x_index])
    lines[:, y_index] = np.apply_along_axis(lambda x: (x-root_y)/spine_len, 0, lines[:,y_index])
    return lines

def kp_horizontal_flip(lines, prob, w):
    x_index = np.arange(0, lines.shape[1], 2)
    if prob < 0.5:
        lines[:,x_index] =  -lines[:, x_index] + w
    return lines

def kp_rotate(line, aug_para):
    degree, prob, w, h = (
        aug_para[0], 
        aug_para[1], 
        aug_para[3], 
        aug_para[4],
    )
    if prob < 0.5:
        center_x, center_y = w / 2, h / 2
        x_index = np.arange(0, line.shape[1], 2)
        y_index = np.arange(1, line.shape[1], 2)
        x = line[:, x_index] - center_x
        y = line[:, y_index] - center_y

        line[:, x_index] = x * cos(degree) + y * sin(degree) + center_x
        line[:, y_index] = -x * sin(degree) + y * cos(degree) + center_y
    return line

def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor

def horizontal_flip(prob, images):
    """
    Perform horizontal flip on the given images and corresponding boxes.
    Args:
        prob (float): probility to flip the images.
        images (tensor): images to perform horizontal flip, the dimension is
            `num frames` x `channel` x `height` x `width`.
        
    Returns:
        images (tensor): images with dimension of
            `num frames` x `channel` x `height` x `width`.
        
    """
    
    if prob < 0.5:
        images = images.flip((-1))
    return images

def frames_rotation(prob, angle, images):

    if prob < 0.5:
        images = T_func.rotate(images, angle=angle)
    return images
        
def frame_augmentation( frames, aug_para):
        frames = horizontal_flip(aug_para[2], frames)
        frames = frames_rotation(aug_para[1], aug_para[0], frames)
        return frames     
