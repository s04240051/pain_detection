from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import numpy as np
import pandas as pd
import os
from lib.utils.logging import get_logger
from . import data_utils as utils
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
        assert self.model_type in ["two_stream",
                                   "rgb", "kp"], f"model type not support"
        self.img_size = cfg.MODEL.IMG_SIZE
        self.clip_len = cfg.CLIP_LENGTH
        self.cfg = cfg

    def label_prepare(self, split):
        file = os.path.join(self.label_file, split+".h5")
        step = self.clip_len if split == "test" else None
        container = []

        df = pd.read_hdf(file, "df_with_missing")
        df = df.applymap(lambda item: int(item))
        data_meter = utils.Dataset_inf(self.cfg)
        for i in range(len(df)):
            line = df.iloc[i]
            extra_label = data_meter.extra_label(line)
            clip_set = self.clip_extract(line, extra_label, step)
            data_meter.label_update(len(clip_set), line["label"], extra_label)
            container.extend(clip_set)
        data_meter.display(logger, split, len(container))
        return container

    def label_update(self, label_info, num_clip, label):
        name_mp = ["not_pain", "pain"]
        key = name_mp[label]
        label_info[key] = label_info[key] + num_clip

    def clip_extract(self, line, extra_label, step=None):
        if not step:
            step = 1 if line["length"] <= 2*self.clip_len else 2

        return [[line.name, i, i+self.clip_len, line["label"]]+extra_label for i in range(line["starts"], line["ends"]-self.clip_len, step)]

    def construct_loader(self, split):
        data_type = self.cfg.DATA.DATA_TYPE
        balance_policy = self.cfg.DATA.BALANCE_POLICY
        if split == "train":
            shuffle = True 
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        clip_df_set = self.label_prepare(split)
        
        sampler = utils.weightedsample(
            clip_df_set, data_type) if balance_policy == 2 else None
        loss_w = [None]*2 if balance_policy == 0 else utils.loss_weight(
            clip_df_set, data_type, balance_policy)
        aug = self.cfg.DATA.AUG if split == "train" else False

        dataset = Pain_dataset(
            clip_df_set,
            self.data_file,
            self.crop_img,
            self.img_size,
            self.model_type,
            self.cfg,
            aug,
        )
        loader = DataLoader(
            dataset,
            self.cfg.RUN.TRAIN_BATCH_SIZE,
            shuffle=(False if sampler else shuffle),
            sampler=sampler,
            num_workers=self.cfg.RUN.NUM_WORKS,
            pin_memory=True,
            drop_last=drop_last,

        )
        return loader, loss_w


class Pain_dataset(Dataset):
    def __init__(
        self,
        label_list,
        video_file,
        crop_img,
        img_size,
        model_type,
        cfg,
        aug
    ):

        self.label_list = label_list
        self.video_file = video_file
        self.crop_img = crop_img

        self.img_size = img_size
        self.data_dict = {}
        self.bbox_col = ["x", "y", "w", "h"]
        self.model_type = model_type
        self.cfg = cfg
        self.aug = aug

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
        aug_para = utils.aug_param_builder(
            clip_inf, video_name) if self.aug else []

        if self.model_type in ["two_stream", "rgb"]:
            frames = self.frames_builder(
                clip_inf[self.bbox_col], video_name, aug_para
            )
            input_data.append(frames)

        if self.model_type in ["two_stream", "kp"]:
            feature = clip_inf.iloc[:, :-4].values
            kps = self.keypoints_builder(
                feature, aug_para
            )
            input_data.append(kps)

        if self.cfg.DATA.DATA_TYPE in ["simple", "diff"]:
            return input_data, int(clip[-1])
        elif self.cfg.DATA.DATA_TYPE == "aux":
            return input_data, [int(clip[-2]), int(clip[-1])]

    def keypoints_builder(self, kp_inf, aug_para):
        if not aug_para:

            return torch.from_numpy(kp_inf.copy()).float()
        kp_inf = utils.kp_horizontal_flip(kp_inf, aug_para[2], aug_para[3])
        kp_inf = utils.kp_rotate(kp_inf, aug_para)
        kp_inf = utils.kp_normal(kp_inf)
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
        frames = utils.tensor_normalize(frames, mean, std)
        frames = frames.permute(0, 3, 1, 2)

        if not aug_para:
            return frames
        else:
            return utils.frame_augmentation(frames, aug_para)
