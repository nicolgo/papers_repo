import os.path
import shutil

import cv2
import glob
from tqdm import tqdm

def secs_to_timestr(secs):
    hrs = secs // (60 * 60)
    min = (secs - hrs * 3600) // 60
    sec = secs % 60
    end = (secs - int(secs)) * 100
    return "{:02d}:{:02d}:{:02d}.{:02d}".format(int(hrs), int(min),
                                                int(sec), int(end))


def clip_specific_videos(raw_video, out_path, duration):
    cap = cv2.VideoCapture(raw_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    left = width/3
    h_size = height - height / 4
    w_size = width - left
    start_sec = 10
    end_sec = start_sec + duration
    cmd = f"ffmpeg -i {raw_video} -vf crop=w={w_size}:h={h_size}:x={left}:y={0},scale=512:512 -ss {secs_to_timestr(start_sec)} -to {secs_to_timestr(end_sec)} -loglevel error {out_path}"
    os.system(cmd)


if __name__ == "__main__":
    video_dir = "D:/disk/data/micro_detect/logger25"
    videos = glob.glob(os.path.join(video_dir, '*'))
    # rename
    print("start to clip...")
    output_dir = "D:/disk/data/micro_detect/clip_small"
    for i in tqdm(range(0, len(videos))):
        raw_video = videos[i]
        out_path = os.path.join(output_dir, f"micro_long_{i}.mp4")
        clip_specific_videos(raw_video, out_path, duration=15)

