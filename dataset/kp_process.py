import pandas as pd
import numpy as np
import itertools
import json
import os
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import KFold

head_set = [
    "L_Eye",
    "R_Eye",
    "L_EarBase",
    "R_EarBase",
    "Nose",
    "Throat",
]
back_set = [
    "TailBase",
    "Withers",
]
leg_set = [
    "L_F_Elbow",
    "L_B_Elbow",
    "R_B_Elbow",
    "R_F_Elbow",
    "L_F_Knee",
    "L_B_Knee",
    "R_B_Knee",
    "R_F_Knee",
    "L_F_Paw",
    "L_B_Paw",
    "R_B_Paw",
    "R_F_Paw",
]

kp_class = (head_set, back_set, leg_set)


def _isArrayLike(obj):

    return not type(obj) == str and hasattr(obj, '__iter__') and hasattr(obj, '__len__')
class keypoints_fix:
    def __init__(self, hdf_file, save_file=None, num_kp=17, need_normal=False):
        if _isArrayLike(hdf_file):
            self.hdf = hdf_file
        elif os.path.isdir(hdf_file):
            self.hdf = [os.path.join(hdf_file, item) for item in os.listdir(
                hdf_file) if item.endswith(".h5")]
        else:
            self.hdf = [hdf_file]
        self.out = save_file
        self.head_set, self.back_set, self.leg_set = kp_class
        self.four_legs = [self.leg_set[i::4] for i in range(4)]
        self.save_file = save_file
        self.need_normal = need_normal
        if need_normal:
            self.new_index = pd.MultiIndex.from_product(
                [
                    ["head1", "head2", "head3"] + self.back_set + self.leg_set,
                    ["x", "y"],
                ],
                names=["bodypart", "coord"]
            )
        else:
            self.new_index = [f"kp{i}" for i in range(2*num_kp)] 

    def pipeline(self, pre_filter=False, threshold=0.3, frame_gap=3):
        for file in tqdm(self.hdf):
            file_name = file.split("\\")[-1]
            if self.save_file:
                out_path = os.path.join(self.save_file, file_name)
                if os.path.isfile(out_path):
                    continue
            kp_df = pd.read_hdf(file, "df_with_missing")

            kp_dfp = self.df_filter(kp_df, pre_filter, threshold, frame_gap)
            out_container = []
            index = kp_dfp.index.values
            line_process_func = self.line_process if self.need_normal else self.line_process_off
            for i in range(len(kp_dfp)):
                out_array = self.kp_fill(kp_dfp.iloc[i], line_process_func)
                if out_array.any():
                    out_container.append(out_array)
                else:
                    index = np.delete(index, i)

            out_df = pd.DataFrame(
                out_container, columns=self.new_index, index=index)
            if self.save_file:
                self.save_hdf(out_df, file_name)

    def save_hdf(self, out_df, file_name):
        hdf_path = os.path.join(self.save_file, file_name)
        csv_file = os.path.join(self.save_file, file_name.strip(".h5")+".csv")

        out_df.to_hdf(
            hdf_path, "df_with_missing", format="table", mode="w"
        )
        out_df.to_csv(csv_file)

    def df_filter(self, kp_df, pre_filter, threshold, frame_gap):
        mask = kp_df.xs("score", level=1, axis=1) > threshold
        kp_dfp = kp_df[mask]
        if pre_filter:
            order_list = (pd.Series(kp_dfp.index)).apply(
                lambda item: int(item.split("_")[-1].strip("0").split(".")[0]))
            kp_dfp = self.first_filter(kp_dfp, order_list, frame_gap)
        delet_set = []

        for i in range(len(kp_df)):
            mask_i = mask.iloc[i]
            if mask_i.sum() < 11 or mask_i[self.back_set].sum() != 2 or \
                    mask_i[self.head_set].sum() < 3 or \
                    mask_i[self.leg_set].sum() < 6:
                delet_set.append(i)
        kp_dfp = kp_dfp.drop(kp_df.index[delet_set])

        return kp_dfp

    def first_filter(self, first_f, order_list, frame_gap):
        # 先相邻帧补全
        full_gap = 2*frame_gap
        for i in range(len(first_f)):
            null_index = np.where(first_f.iloc[i].xs("x", level=1).isnull())[0]
            clip_start = np.clip([i-frame_gap], 0, len(first_f)-full_gap)[0]
            frame_order = order_list[i]

            if null_index.any():
                for index in null_index:
                    item_deter = ~(
                        first_f.iloc[clip_start:clip_start+full_gap, 3*index].isnull()).values

                    frame_deter = (order_list[clip_start:clip_start+full_gap].values >= (frame_order-frame_gap))*(
                        order_list[clip_start:clip_start+full_gap].values <= (frame_order+frame_gap))
                    deter_list = np.where(item_deter * frame_deter)[0]

                    if deter_list.any():
                        deter_index = deter_list[len(deter_list) // 2]
                        first_f.iloc[i, 3*index] = first_f.iloc[
                            clip_start + deter_index, 3*index
                        ]

                        first_f.iloc[i, 3*index+1] = first_f.iloc[
                            clip_start + deter_index, 3*index+1
                        ]
        return first_f

    def kp_fill(self, example_line, line_process):
        # 补腿
        line_x = example_line.xs("x", level=1)
        line_y = example_line.xs("y", level=1)
        temp_set = defaultdict(list)
        for leg in self.four_legs:
            notnan = (~np.isnan(line_x[leg])).sum()
            temp_set[notnan].append(leg)

        if len(temp_set[3]) == 4:
            return line_process((line_x, line_y))
        if temp_set[2]:
            for leg in temp_set[2]:
                line_x[leg], line_y[leg] = self.fix_two(
                    line_x, line_y, leg)

        case3 = temp_set[0]+temp_set[1]
        if case3:
            for leg in case3:
                leg_index = self.four_legs.index(leg)
                vertical_pare = 1-leg_index if leg_index < 1.5 else 5-leg_index
                hori_pare = 3-leg_index
                if temp_set[3]:
                    if self.four_legs[hori_pare] in temp_set[3]:
                        line_x[leg], line_y[leg] = self.hori_shift(
                            (line_x, line_y), self.four_legs[hori_pare])
                    elif self.four_legs[vertical_pare] in temp_set[3]+temp_set[2]:
                        line_x[leg], line_y[leg] = self.vertical_shift(
                            (line_x, line_y), leg, self.four_legs[vertical_pare], self.back_set)
                    else:
                        line_x[leg], line_y[leg] = self.vertical_shift(
                            (line_x, line_y), leg, temp_set[3][0], self.back_set)
                else:
                    if self.four_legs[hori_pare] in temp_set[2]:
                        line_x[leg], line_y[leg] = self.hori_shift(
                            (line_x, line_y), self.four_legs[hori_pare])
                    else:
                        line_x[leg], line_y[leg] = self.vertical_shift(
                            (line_x, line_y), leg, temp_set[2][0], self.back_set)

        return line_process((line_x, line_y))

    def line_process(self, linexy):
        # 以脊椎的中点为坐标原点，得相对坐标。 除以脊椎长度标准化
        out_contain = []
        linex, liney = linexy
        spine_len = np.sqrt(
            (linex[back_set][0]-linex[back_set][1])**2 +
            (liney[back_set][0]-liney[back_set][1])**2
        )
        if spine_len == 0:
            return np.array(out_contain)
        for line in linexy:
            # normalize
            root = (line[back_set][0]+line[back_set][1])/2
            line = line.apply(lambda item: item-root)
            line = (line[~np.isnan(line)].values)/spine_len
            out_contain.append(np.append(line[:3], line[-14:]))
        out_contain = np.array(out_contain)
        return out_contain.flatten(order="F")

    def line_process_off(self, linexy):
        # 不处理相对坐标和标准化， 只进行筛选，记录脊椎长度，返回的数据多一列
        out_contain = []
        linex, liney = linexy
        spine_len = np.sqrt(
            (linex[back_set][0]-linex[back_set][1])**2 +
            (liney[back_set][0]-liney[back_set][1])**2
        )
        if spine_len == 0:
            return np.array(out_contain)
        for line in linexy:

            line = line[~np.isnan(line)].values
            out_contain.append(np.append(line[:3], line[-14:]))
            #out_contain.append(line)
        out_contain = np.array(out_contain).flatten(order="F")
        return out_contain

    def fix_two(self, line_x, line_y, leg):
        # 缺一个点补全
        leg_x, leg_y, backx, backy = line_x[leg], line_y[leg], line_x[self.back_set], line_y[self.back_set]
        nan_index = np.where(np.isnan(leg_x))[0][0]
        xb, yb = (backx[0], backy[0]) if leg[0].split(
            "_")[1] == "B" else (backx[1], backy[1])
        if nan_index == 0:
            leg_x[0] = (leg_x[1] + xb)/2
            leg_y[0] = (leg_y[1]+yb)/2

        elif nan_index == 1:
            leg_x[1] = (leg_x[0] + leg_x[2])/2
            leg_y[1] = (leg_y[0] + leg_y[2])/2

        else:
            leg_x[2] = 2*leg_x[1]-leg_x[0]
            leg_y[2] = 2*leg_y[1]-leg_y[0]
        return leg_x, leg_y

    def hori_shift(self, xy, leg_shift):
        # 横向遮挡补全
        linex, liney = xy

        return ((linex[leg_shift]+np.random.randint(10, 20)).to_list(),
                (liney[leg_shift]+np.random.randint(10, 20)).to_list())

    def vertical_shift(self, xy, leg, leg_shift, back_set):
        # 纵向遮挡补全
        linex, liney = xy

        F = True if leg[0].split("_")[1] == "F" else False
        if F:
            shift_vector = (linex[back_set][1]-linex[back_set]
                            [0], liney[back_set][1]-liney[back_set][0])
            return ((linex[leg_shift]+shift_vector[0]*0.75).to_list(),
                    (liney[leg_shift]+shift_vector[1]*0.75).to_list())
        else:
            shift_vector = (linex[back_set][0]-linex[back_set]
                            [1], liney[back_set][0]-liney[back_set][1])
            return ((linex[leg_shift]+shift_vector[0]*0.75).to_list(),
                    (liney[leg_shift]+shift_vector[1]*0.75).to_list())


if __name__ == "__main__":
    
    hdf_file = r"D:\pose\pain\data\pain_data\annotation\keypoints"
    save_file = r"D:\pose\pain\data\pain_data\annotation\fixed_kp_raw_full"
    data_model = keypoints_fix(hdf_file, save_file, num_kp=20)
    data_model.pipeline(pre_filter=True)
    '''
    hdf_file = r"D:\pose\pain\data\pain_data\annotation\fixed_kp_raw"
    save_file = r"D:\pose\pain\data\pain_data\annotation\kp_valid_raw"
    data_model = Dataset_builder(hdf_file, save_file = save_file, split_file=r"D:\pose\pain\data\pain_data\annotation\split_nonormal")
    data_model.run(gap_len=9, clip_len=8,k_fold=5,default_split=r"D:\pose\pain\data\pain_data\annotation\split")
    '''