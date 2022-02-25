import os
import cv2
import imageio
import numpy as np
import pandas as pd

class Flow_image_builder:
    def __init__(
        self,
        save_file,
        split_path = r"D:\pose\pain\data\pain_data\annotation\split",
        crop_path = r"D:\pose\pain\data\pain_data\crop_img",
        kp_path = r"D:\pose\pain\data\pain_data\annotation\kp_valid", 
        
    ):
        hdf_fils = [os.path.join(split_path,item) for item in os.listdir(split_path) if item.endswith(".h5")]
        hdf_set = [pd.read_hdf(item, "df_with_missing") for item in hdf_fils]
        self.all_clips = pd.concat(hdf_set, axis=0)
        self.video_set = self.all_clips.index.unique()
        self.crop_path = crop_path
        self.kp_path = kp_path
        self.save_file = save_file
    
    def pipeline(self):
        for video in self.video_set:
            clip_set = self.all_clips.loc[video]
            video_hdf = pd.read_hdf(os.path.join(self.kp_path, video), "df_with_missing")
            raw_name = video.split(".")[0]
            crop_file = os.path.join(self.crop_path, raw_name)
            save_video_path = os.path.join(self.save_file, raw_name)
            if not os.path.isdir(save_video_path):
                os.makedirs(save_video_path)
            else:
                continue
            video_name = video_hdf.index
            if type(clip_set) is pd.core.series.Series:
                self.clip_process(clip_set, video_name, crop_file, save_video_path)
            else:
                for index in range(len(clip_set)):
                    line =  clip_set.iloc[index]
                    self.clip_process(line, video_name, crop_file, save_video_path)
        
    def clip_process(self, line, video_name, crop_file, save_video_path):
        starts, ends = line["starts"], line["ends"]
        seq_name = video_name[starts:ends]
        name_pair = [(seq_name[i], seq_name[i+1]) for i in range(len(seq_name)-1)]
        for i, (pre, next) in enumerate(name_pair):
            pair = (
                os.path.join(crop_file, pre),
                os.path.join(crop_file, next),
            )
            flow_img = self.draw_flow(pair)

            self.save_flow(next, save_video_path, flow_img)
            if i == 0:
                self.save_flow(pre, save_video_path, flow_img)

            
    def save_flow(self, path,save_file, img):
        img_path = os.path.join(save_file, path)
        imageio.imwrite(img_path, img)
    
    def draw_flow(self, pair):
        img_path1, img_path2 = pair
        pre = cv2.imread(img_path1)
        next = cv2.imread(img_path2)

        pre = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY)
        next = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(pre, next, flow=None, pyr_scale=0.5, levels=3,
                                        winsize=15, iterations=3, poly_n=5,
                                        poly_sigma=1.2, flags=0)
        extra_channel =self.get_flow_magnitude(flow)
        flow_img1 = np.concatenate((flow, extra_channel), axis=2)
        flow_img1 = cv2.normalize(flow_img1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return flow_img1
    
    def get_flow_magnitude(self, flow):
        """
        Compute the magnitude of the optical flow at every pixel.
        :param flow: np.ndarray [width, height, 2]
        :return: np.ndarray [width, height, 1]
        """
        magnitude = np.sqrt(np.power(flow[:, :, 0], 2) + np.power(flow[:, :, 1], 2))
        magnitude = np.reshape(magnitude, (flow.shape[0], flow.shape[1], 1))
        return magnitude

if __name__ =="__main__":
    data_model = Flow_image_builder(r"D:\pose\pain\data\pain_data\annotation\optical_flow")
    data_model.pipeline()