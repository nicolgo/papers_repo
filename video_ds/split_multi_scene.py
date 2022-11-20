import os.path
import shutil

import cv2
import glob
from tqdm import tqdm

if __name__ == "__main__":
    video_dir = "D:/disk/data/sad_new/all_sad_videos"
    videos = glob.glob(os.path.join(video_dir, '*'))

    for i in tqdm(range(0, len(videos))):
        raw_video = videos[i]
        print(f"clip the {raw_video} ....")
        cmd = f"scenedetect --input {raw_video} detect-content split-video"
        os.system(cmd)
