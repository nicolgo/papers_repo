import json
import os
import os.path as osp
import shutil

from pytube import YouTube


def get_video_id(file_path):
    with open(file_path) as f:
        data_dict = json.load(f)
    for x in data_dict:
        for value in data_dict[x]:
            yield value


def main():
    json_file = "search_res.json"
    os.makedirs("D:/disk/data/demo", exist_ok=True)
    os.makedirs("D:/disk/data/demo/video", exist_ok=True)
    for id in get_video_id(json_file):
        video_save_path = "D:/disk/data/demo/video"
        yt = YouTube(f'https://www.youtube.com/watch?v={id}')
        yt.streams.get_highest_resolution().download(video_save_path, filename=f"{id}.mp4")
        # shutil.move(osp.join(video_save_path, os.listdir(video_save_path)[0]), osp.join(video_save_path, f"{id}.mp4"))


if __name__ == '__main__':
    main()
