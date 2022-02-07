import json
import torch
from PIL import Image
import os
import numpy as np
from collections import Counter
import torchvision.transforms.functional as T_func
from torch.utils.data import WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
from math import cos, sin


def aug_param_builder(clip_inf, video_name):
    path = os.path.join(video_name, clip_inf.index[0])
    img = Image.open(path)
    img_w, img_h = img.size
    rotate_degree = np.random.randint(-180, 180)
    rotate_prob, flip_prob = np.random.uniform(), np.random.uniform()
    aug_para = [
        rotate_degree, rotate_prob, flip_prob, img_w, img_h
    ]
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

    lines[:, x_index] = np.apply_along_axis(
        lambda x: (x-root_x)/spine_len, 0, lines[:, x_index])
    lines[:, y_index] = np.apply_along_axis(
        lambda x: (x-root_y)/spine_len, 0, lines[:, y_index])
    return lines


def kp_horizontal_flip(lines, prob, w):
    x_index = np.arange(0, lines.shape[1], 2)
    if prob < 0.5:
        lines[:, x_index] = -lines[:, x_index] + w
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


def frame_augmentation(frames, aug_para):
    frames = horizontal_flip(aug_para[2], frames)
    frames = frames_rotation(aug_para[1], aug_para[0], frames)
    return frames


class Dataset_inf:
    def __init__(self, cfg):
        if not cfg.DATA.REQUIRE_AUX:
            self.label_info = {"pain": 0, "not_pain": 0}
            self.two_label = 0
        elif cfg.DATA.EXTRA_LABEL:
            self.label_info = {
                "pain": 0, "not_pain": 0, "neural_pain": 0, "orth_pain": 0
            }
            self.two_label = 1
        elif os.path.isfile(cfg.AUXILIARY_FILE):
            self.label_info = {
                "pain": 0, "not_pain": 0, "neural_pain": 0, "orth_pain": 0
            }
            with open(cfg.AUXILIARY_FILE) as f:
                self.video_set = json.load(f)
            self.two_label = 2
        else:
            raise FileNotFoundError("data type define unclear")

        self.name_map = ["not_pain", "pain"]
        self.name_map2 = ["not_pain", "neural_pain", "orth_pain"]

    def extra_label(self, line):
        name = line.name
        if self.two_label == 0:
            return []
        elif self.two_label == 1:
            return line["extra_label"]
        if name in self.video_set["neural_pain"]:
            return [1]
        elif name in self.video_set["orth_pain"]:
            return [2]
        else:
            return [0]

    def label_update(self, num_clip, label, extra_label):

        key = self.name_map[label]
        self.label_info[key] = self.label_info[key] + num_clip
        if extra_label and extra_label[0] != 0:
            key = self.name_map2[extra_label[0]]
            self.label_info[key] = self.label_info[key] + num_clip

    def display(self, logger, split, num):
        self.label_info[split] = num
        label_item = [f"{key}: {item}" for key,
                      item in self.label_info.items()]
        label_item = ",".join(label_item)
        logger.info(label_item)


def weightedsample(clip_inf, data_type):
    clip_inf = np.array(clip_inf)
    label_list = clip_inf[:, -2] if data_type == "aux" else clip_inf[:, -1]
    weight_count = {key: 1./value for key, value in Counter(label_list).items()}
    sample_weight = torch.tensor([weight_count[item] for item in label_list])
    sampler = WeightedRandomSampler(
        sample_weight, len(sample_weight), replacement=True)
    return sampler


def loss_weight(clip_inf, data_type, balance_policy):
    clip_inf = np.array(clip_inf)
    if data_type in ["simple", "diff"]:
        if balance_policy == 2:
            return [None]
        else:
            labels = clip_inf[:, -1]
            weight = compute_class_weight(
                "balanced", classes=np.unique(labels), y=labels
            )
            return [torch.tensor(weight,dtype=torch.float)]
    else:
        out_weight = []
        label_set = clip_inf[:,-2:].T
        for label in label_set:
            weight = compute_class_weight(
                "balanced", classes=np.unique(label), y=label
            )
            out_weight.append(torch.tensor(weight,dtype=torch.float))
        
        weight_count = Counter(label_set[0])
        third_weight = out_weight[1].clone().detach()
        factor = weight_count["0"]/weight_count["1"]
        third_weight[0] *= factor
        third_weight = torch.tensor(third_weight,dtype=torch.float)
        out_weight.append(third_weight)
        return out_weight[:2] if balance_policy == 1 else [None, out_weight[2]]