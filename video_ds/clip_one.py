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


def clip_specific_videos(raw_video, out_path, duration, start_time):
    cap = cv2.VideoCapture(raw_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = height - height / 7
    left = (width - size) / 2
    start_sec = start_time
    end_sec = start_sec + duration
    # cmd = f"ffmpeg -i {raw_video} -vf crop=w={size}:h={size}:x={left}:y={0},scale=512:512 -ss {secs_to_timestr(start_sec)} -to {secs_to_timestr(end_sec)} -loglevel error {out_path}"
    cmd = f"ffmpeg -i {raw_video} -vf scale=512:512 -ss {secs_to_timestr(start_sec)} -to {secs_to_timestr(end_sec)} -loglevel error {out_path}"
    os.system(cmd)


if __name__ == "__main__":
    video_dir = "D:/disk/data/space"
    videos = glob.glob(os.path.join(video_dir, '*'))
    output_dir = "D:/disk/data/surprised"
    j = 0
    for i in tqdm(range(0, len(videos))):
        raw_video = videos[i]
        for i in tqdm(range(0, 2400, 60)):
            out_path = os.path.join(output_dir, f"surprised_{j}.mp4")
            clip_specific_videos(raw_video, out_path, duration=15, start_time=i)
            j = j + 1
            # i = i + 60
