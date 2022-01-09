import pandas as pd
import numpy as np
import itertools
import os
from tqdm import tqdm
from collections import defaultdict

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


class Dataset_builder:
    def __init__(
        self,
        hdf_file,
        save_file=None,
        video_file=r"D:\pose\Eksik OFTler\OneDrive-2021-10-28",
        crop_file=r"D:\pose\pain\data\annotation\crop",
        split_file=r"D:\pose\pain\data\annotation\split",
        num_kp=17,
    ):
        assert os.path.isdir(hdf_file)
        self.hdf = [item for item in os.listdir(
            hdf_file) if item.endswith(".h5")]
        self.dir = hdf_file
        self.out = save_file
        self.crop_file = crop_file
        self.split_file = split_file
        self.painset = [item.split(".")[0]+".h5"
                        for item in os.listdir(video_file+"\\pain")]
        self.notpainset = [item.split(".")[0]+".h5"
                           for item in os.listdir(video_file+"\\not_pain")]
        self.img_inf_col = ["x", "y", "w", "h"]
        self.data_col = [f"kp{i}" for i in range(2*num_kp)]

    def pipeline(self, gap_len=9, clip_len=5):
        split_set = self.train_test_split()
        split_df_set = []
        for i, hdfs in enumerate(split_set):
            df_temp = []
            is_pain = True if i % 2 == 0 else False
            for file in hdfs:
                test_hdf = pd.read_hdf(
                    os.path.join(self.dir, file), "df_with_missing"
                )
                label, data = self.range_select(test_hdf, file, is_pain, gap_len, clip_len)
                df_temp.append(label)
                self.save_data(data, file)
            split_df = pd.concat(df_temp, axis=0)
            split_df_set.append(split_df)

        inf = zip(
            ("train", "test"), (split_df_set[:2], split_df_set[2:])
        )
        self.save_label(inf)

    def save_label(self, inf):
        for name, df in inf:
            df = pd.concat(df, axis=0)
            df.to_hdf(
                os.path.join(self.split_file, name+".h5"), "df_with_missing", format="table", mode="w"
            )
            df.to_csv(os.path.join(self.split_file, name+".csv"))

    def save_data(self, video_data, name):

        video_data.to_hdf(
            os.path.join(self.out, name), "df_with_missing", format="table", mode="w"
        )
        video_data.to_csv(os.path.join(
            self.out, name.strip(".h5")+".csv"))

    def train_test_split(self, rate=0.2):
        pain = list(set(self.hdf) & set(self.painset))
        notpain = list(set(self.hdf) & set(self.notpainset))
        test_pain = np.random.choice(pain, round(
            len(pain)*rate), replace=False).tolist()
        train_pain = list(set(pain) - set(test_pain))
        test_not = np.random.choice(notpain, round(
            len(notpain)*rate), replace=False).tolist()
        train_not = list(set(notpain) - set(test_not))

        return [train_pain, train_not, test_pain, test_not]

    def range_select(self, test_hdf, name, is_pain, gap_len=9, clip_len=5):
        order_list = (pd.Series(test_hdf.index)).apply(
            lambda item: int(item.split("_")[-1].strip("0").split(".")[0]))
        gap = np.append(order_list.values[1:], 0)-order_list.values
        gap[-1] = -gap[-1]
        end_point = np.arange(len(gap))[gap > gap_len]
        start_point = np.append([0], end_point[:-1])
        ends = end_point[(end_point-start_point) > clip_len] + 1
        starts = start_point[(end_point-start_point) > clip_len] + 1
        starts[0] = 0 if starts[0] == 1 else starts[0]
        length = ends-starts

        clip_index = np.concatenate([list(range(s, e))
                                    for s, e in zip(starts, ends)])
        index_list = test_hdf.index[clip_index]
        crop_inf = pd.read_hdf(os.path.join(self.crop_file, name))
        crop_inf = (crop_inf.loc[index_list])[self.img_inf_col]

        clip_df = test_hdf.iloc[clip_index]
        clip_df.columns = self.data_col
        clip_df = pd.concat([clip_df, crop_inf], axis=1)
        end_new = list(itertools.accumulate(length))
        start_new = [0] + end_new[:-1]
        index = [name]*len(start_new)
        label = [1]*len(start_new) if is_pain else [0]*len(start_new)

        return pd.DataFrame({"starts": start_new, "ends": end_new, "length": length, "label": label}, index=index), clip_df


class keypoints_fix:
    def __init__(self, hdf_file, save_file=None):
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
        self.new_index = pd.MultiIndex.from_product(
            [
                ["head1", "head2", "head3"] + self.back_set + self.leg_set,
                ["x", "y"],
            ],
            names=["bodypart", "coord"]
        )

    def pipeline(self, pre_filter=False, threshold=0.3, frame_gap=3):
        for file in tqdm(self.hdf):
            file_name = file.split("\\")[-1]
            if self.out:
                out_path = os.path.join(self.save_file, file_name)
                if os.path.isfile(out_path):
                    continue
            kp_df = pd.read_hdf(file, "df_with_missing")

            kp_dfp = self.df_filter(kp_df, pre_filter, threshold, frame_gap)
            out_container = []
            index = kp_dfp.index.values
            for i in range(len(kp_dfp)):
                out_array = self.kp_fill(kp_dfp.iloc[i])
                if out_array.any():
                    out_container.append(out_array)
                else:
                    index = np.delete(index, i)

            out_df = pd.DataFrame(
                out_container, columns=self.new_index, index=index)
            if self.out:
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

    def kp_fill(self, example_line):
        # 补腿
        line_x = example_line.xs("x", level=1)
        line_y = example_line.xs("y", level=1)
        temp_set = defaultdict(list)
        for leg in self.four_legs:
            notnan = (~np.isnan(line_x[leg])).sum()
            temp_set[notnan].append(leg)

        if len(temp_set[3]) == 4:
            return self.line_process((line_x, line_y))
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

        return self.line_process((line_x, line_y))

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
    '''
    hdf_file = r"D:\pose\pain\data\annotation\keypoints"
    save_file = r"D:\pose\pain\data\annotation\fixed_kp"
    data_model = keypoints_fix(hdf_file, save_file)
    data_model.pipeline(pre_filter=True)
    '''
    hdf_file = r"D:\pose\pain\data\annotation\fixed_kp"
    save_file = r"D:\pose\pain\data\annotation\kp_valid"
    data_model = Dataset_builder(hdf_file, save_file)
    data_model.pipeline(gap_len=9, clip_len=8)
