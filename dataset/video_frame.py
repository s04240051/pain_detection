import cv2
import os
import numpy as np
from PIL import Image
from subprocess import call


def _isArrayLike(obj):

    return not type(obj) == str and hasattr(obj, '__iter__') and hasattr(obj, '__len__')


def cv_frame(input_dir, video_list, out_dir, start=None, scale=0.2, sample_rate=2, sample_sec=60):

    if not start:
        start = [[30, 165, 270] for _ in range(len(video_list))]
    else:
        assert len(start) == len(video_list)
    if _isArrayLike(sample_sec):
        assert len(sample_sec) == len(video_list)
    for i, video in enumerate(video_list):
        input_video = os.path.join(input_dir, video)
        name = video.split(".")[0]
        out_file = os.path.join(out_dir, name)
        if not os.path.isdir(out_file):
            os.makedirs(out_file)
            cap = cv2.VideoCapture(input_video)

            fps = int(cap.get(cv2.CAP_PROP_FPS))
            start_step = list(map(lambda x: x*fps, start[i]))
            select_step = set()
            step = fps//sample_rate
            if not _isArrayLike(sample_sec):
                sample_nums = [sample_sec*sample_rate]*len(start_step)
            else:
                assert len(sample_sec[i]) == len(start[i])
                sample_nums = [sec * sample_rate for sec in sample_sec[i]]
            for j, item in enumerate(start_step):
                select_step = select_step.union(
                    set(range(item, item+step*sample_nums[j], step)))

            select_len = len(select_step)
            read, img = cap.read()
            count_frame = 0

            count_sample = 0
            while read:
                if count_frame in select_step:
                    if scale:
                        img = cv2.resize(
                            img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA
                        )
                    target_file = os.path.join(
                        out_file, name+"_%04d.jpg" % count_sample)  # f"{name}_{count_sample}.jpg")
                    cv2.imencode('.jpg', img)[1].tofile(target_file)
                    count_sample += 1
                if count_sample >= select_len:
                    break
                read, img = cap.read()
                count_frame += 1
        cap.release()


'''
video_dir = r"D:\pose\Eksik OFTler\OneDrive-2021-10-28\pain"
target_dir = r"D:\pose\pain\data\raw_frame"
video_list = os.listdir(video_dir)
#video_list = [item for item in video_list if item.endswith(".mp4")]
video_list = ["pasa_side.mp4"]
video_clip = [
    ("00:00:36", "00:10:25")
    ]
'''


def ffmpeg_cut(video_dir, target_dir, video_clip=None, video_list=None):
    if not video_list:
        video_list = os.listdir(video_dir)
    if video_clip:
        assert len(video_clip) == len(video_list)
        manual = True
    else:
        manual = False
    for i, video in enumerate(video_list):
        input_video = os.path.join(video_dir, video)

        name = video.split(".")[0]
        out_file = os.path.join(target_dir, name)
        if not os.path.isdir(out_file):
            os.makedirs(out_file)
            if manual:
                start, end = video_clip[i]
            else:
                start, end = get_duration(input_video)

            out_frame = os.path.join(out_file, f"{name}_%04d.jpg")
            """
            call(["ffmpeg", '-i', input_video, "-r", "2", "-s",
                "676,507", "-ss", start, "-to", end, out_frame])
            """
            call(["ffmpeg", '-i', input_video, "-r", "3",
                 "-ss", start, "-to", end, out_frame])


def get_duration(input_video):
    cap = cv2.VideoCapture(input_video)
    if cap:
        fps = cap.get(5)
        num_frames = cap.get(7)
        seconds = int(num_frames/fps)-60
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        end = "%02d:%02d:%02d" % (h, m, s)
        start = "00:00:30"
        return start, end


if __name__ == "__main__":
    video_dir = r"D:\pose\Eksik OFTler\OneDrive-2021-10-28\not_pain"
    video_list = ["Mafi-oft-ilk-cam1_Trim.mp4"]#os.listdir(video_dir)
    target_dir = r"D:\pose\pain\data\raw_frame"
    ffmpeg_cut(video_dir, target_dir, video_list=video_list)
