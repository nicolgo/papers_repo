import glob
import os.path
import shutil

import cv2
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
    size = height * 0.9
    left = (width - size) / 2
    start_sec = start_time
    end_sec = start_sec + duration
    # cmd = f"ffmpeg -i {raw_video} -vf crop=w={size}:h={size}:x={left}:y={0},scale=512:512 -ss {secs_to_timestr(start_sec)} -to {secs_to_timestr(end_sec)} -loglevel error {out_path}"
    cmd = f"ffmpeg -i {raw_video} -vf crop=w={size}:h={size}:x={left}:y={0},scale=512:512 -ss {secs_to_timestr(start_sec)} -to {secs_to_timestr(end_sec)} -loglevel error {out_path}"
    os.system(cmd)


if __name__ == "__main__":
    video_dir = "D:/disk/data/excited_orig"
    videos = glob.glob(os.path.join(video_dir, '*'))
    # rename
    rename_dir = "D:/disk/data/excited_orig/rename"
    # for i in tqdm(range(0, len(videos))):
    #     raw_video = videos[i]
    #     shutil.copy(raw_video, os.path.join(rename_dir, f"excited_rename_{i}.mp4"))
    rename_videos = glob.glob(os.path.join(rename_dir, '*'))
    print("start to clip...")
    output_dir = "D:/disk/data/excited_clips"
    for i in tqdm(range(0, len(rename_videos))):
        raw_video = rename_videos[i]
        # create video capture object
        data = cv2.VideoCapture(raw_video)
        frames = data.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = data.get(cv2.CAP_PROP_FPS)
        seconds = round(frames / fps) - 15
        for j in range(0, seconds, 10):
            out_path = os.path.join(output_dir, f"excited_{i}_{j}.mp4")
            clip_specific_videos(raw_video, out_path, duration=15, start_time=j)
