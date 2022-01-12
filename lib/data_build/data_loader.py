from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import torch.nn as nn
from PIL import Image
import torch
import numpy as np
import pandas as pd
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
        
        self.img_size = cfg.MODEL.IMG_SIZE
        self.clip_len = cfg.CLIP_LENGTH
    
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

    def construct_loader(self, batch_size, num_works, split):
        if split == "train":
            shuffle = True
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        clip_df_set = self.label_prepare(split)
        dataset = Pain_dataset(clip_df_set, self.data_file,
                               self.crop_img, self.img_size)
        loader = DataLoader(
            dataset,
            batch_size,
            shuffle,
            num_workers=num_works,
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
        img_size
    ):

        self.label_list = label_list
        self.video_file = video_file
        self.crop_img = crop_img

        self.img_size = img_size
        self.data_dict = {}
        self.bbox_col = ["x", "y", "w", "h"]

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
        
        clip_inf = video_df.iloc[clip[1]:clip[2]]
        frames = self.frames_builder(
            clip_inf[self.bbox_col], os.path.join(
                self.crop_img, file_name.split(".")[0])
        )
        feature  = clip_inf.iloc[:, :-4].values
        kps = torch.from_numpy(feature.copy()).float()
        label = clip[-1]
        return [frames, kps], int(label)

    def frames_builder(self, frame_inf, video_name):
        crop_threshold = [300,  450, 600]
        frames = []
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
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
        return frames


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

