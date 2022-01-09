import torch
import cv2
import os
from PIL import Image
import pandas as pd
import numpy as np
import subprocess
from tqdm import tqdm

def _isArrayLike(obj):

    return not type(obj) == str and hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class Databuilder:
    def __init__(self, img_file, save_root, width, height, load_model=True):
        self.img = img_file
        self.save_root = save_root
        self.crop_w = width/2
        self.crop_h = height/2
        self.setup(load_model)

    def setup(self, load_model):
        self.crop_save = os.path.join(self.save_root, "annotation\\crop")
        self.keypoints_save = os.path.join(
            self.save_root, "annotation\\keypoints")
        self.img_save = os.path.join(self.save_root, "crop_img")
        self.keypoints_vis = os.path.join(self.save_root, "keypoint_img")
        if load_model:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5x')
            model.conf = 0.01  
            model.iou = 0.45 
            model.classes = [16]
            model.multi_label = False
            self.model = model

    def bbox_crop(self, img_dir):
        img_list = [os.path.join(img_dir, img) for img in os.listdir(
            img_dir) if img.endswith(".jpg")]
        dog_name = img_dir.split("\\")[-1]

        bbox = pd.DataFrame(
            columns=["xmin", "ymin", "xmax", "ymax", "confidence", "class", "name"])
        empty_name = []

        for i, img in enumerate(tqdm(img_list)):
            results = self.model(img)
            df_bb = results.pandas().xyxy[0]
            if df_bb.empty:
                empty_name.append(i)
            else:

                bbox = bbox.append(df_bb.iloc[0]) if len(df_bb) == 1 else bbox.append(
                    df_bb.loc[df_bb["confidence"].idxmax()])
        order_list = np.arange(len(img_list))
        order_list = np.delete(order_list, empty_name)
        order_list += 1
        img_list = np.asarray(img_list)
        img_list = np.delete(img_list, empty_name)
        
        img_name = list(
            map(
                lambda item: 
                     item.split("\\")[-1], img_list
            )
        )
        
        im = Image.open(img_list[0])
        img_w, img_h = im.size
        bbox.index = img_name
        
        bboxcrop = self.table_process(bbox, img_w, img_h)
        bboxcrop["order"] = order_list
        return bboxcrop

    def table_process(self, bbox, img_w, img_h):
        # crop 坐标
        bbox_coord = bbox.iloc[:, :4]
        bbox_coord["xcrop"] = (
            (bbox_coord["xmin"]+bbox_coord["xmax"])/2)-self.crop_w
        bbox_coord["ycrop"] = (
            (bbox_coord["ymin"]+bbox_coord["ymax"])/2)-self.crop_h
        bbox_coord["xcrop"] = np.clip(
            bbox_coord["xcrop"], 0, img_w-self.crop_w*2)
        bbox_coord["ycrop"] = np.clip(
            bbox_coord["ycrop"], 0, img_h-self.crop_h*2)

        out_dict = {
            "x": (bbox_coord["xmin"]-bbox_coord["xcrop"]).values,
            "y": (bbox_coord["ymin"]-bbox_coord["ycrop"]).values,
            "w": (bbox_coord["xmax"]-bbox_coord["xmin"]).values,
            "h": (bbox_coord["ymax"]-bbox_coord["ymin"]).values,

        }
        # bbox坐标扩大10%
        out_dict_extend = {
            "x": np.clip(out_dict["x"]-out_dict["w"]*0.1/2, 0, self.crop_w*2),
            "y": np.clip(out_dict["y"]-out_dict["h"]*0.1/2, 0, self.crop_h*2),
            "w": np.clip(out_dict["w"]*1.1, 0, self.crop_w*2),
            "h": np.clip(out_dict["h"]*1.1, 0, self.crop_h*2),
        }
        out_inf_extend = pd.DataFrame(out_dict_extend)
        out_inf_extend.index = bbox_coord.index
        # 合成一个表
        result = pd.concat([bbox_coord, out_inf_extend],
                           axis=1)
        result = result.apply(lambda item: round(item), axis=0)
        return result

    def save_ann(self, bboxcrop, name):
        hdf_file = os.path.join(self.crop_save, name+".h5")
        csv_file = os.path.join(self.crop_save, name+".csv")
        if not os.path.isfile(hdf_file):
            bboxcrop.to_hdf(
                hdf_file, "df_with_missing", format="table", mode="w"
            )
            bboxcrop.to_csv(csv_file)
            

    def save_img(self, crop_df, input_dir, out_dir):
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
            for i, name in enumerate(crop_df.index):
                img = Image.open(
                    os.path.join(input_dir, name.split("\\")[-1])
                )
                x, y = crop_df["xcrop"][i], crop_df["ycrop"][i]
                img = img.crop((x, y,
                                x+self.crop_w*2, y+self.crop_h*2)
                               )
                img.save(
                    os.path.join(out_dir, name.split("\\")[-1])
                )

    def run_pose(self, hdf_file, crop_img, pose_vis):
        if os.path.isdir(crop_img) and os.path.isfile(hdf_file):
            command = [
                "python",
                "demo/top_down_img_custom.py ",
                "configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/animalpose/hrnet_w32_animalpose_256x256.py",
                "checkpoints/hrnet_w32_animalpose_256x256-1aa7f075_20210426.pth",
                "--img-root",
                crop_img,
                "--hdf-file",
                hdf_file,
                "--out-df-root",
                self.keypoints_save,
            ]
            if pose_vis:
                name = crop_img.split("\\")[-1]
                pose_img = os.path.join(self.keypoints_vis, name)
                if not os.path.isdir(pose_img):
                    os.makedirs(pose_img)
                command += ["--out-img-root", pose_img]
            subprocess.run(command)

    def pipeline(self, needcrop=True, pose=True, pose_vis=True):
        images = self.img if _isArrayLike(self.img) else [self.img]
        cur_dir = os.getcwd()
        for img_dir in images:
            name = img_dir.split("\\")[-1]
            crop_img = os.path.join(self.img_save, name)
            if needcrop:
                if os.path.isdir(crop_img):
                    continue
                bboxcrop = self.bbox_crop(img_dir)
                
                self.save_ann(bboxcrop, name)
                self.save_img(bboxcrop, img_dir, crop_img)

            if pose:
                hdf_file = os.path.join(self.crop_save, name+".h5")
                pose_hdf = os.path.join(self.keypoints_save, name+".h5")
                if os.path.isfile(pose_hdf):
                    continue
                if os.path.isfile(hdf_file):
                    mmpose_dir = r"D:\pose\mmpose\mmpose"
                    os.chdir(mmpose_dir)
                    self.run_pose(hdf_file, crop_img, pose_vis)
                    os.chdir(cur_dir)

if __name__ == "__main__":
    img_file = r"D:\pose\pain\data\raw_frame"
    
    img = ["Mafi-oft-ilk-cam1_Trim"]
    #img_file_list = [os.path.join(img_file, item) for item in os.listdir(img_file)]
    img_file_list = [os.path.join(img_file, item) for item in img]
    save_file = r"D:\pose\pain\data"
    data_model = Databuilder(img_file_list, save_file, 600, 600, load_model=True)
    data_model.pipeline(needcrop=True, pose=True)