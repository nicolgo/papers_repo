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
    json_file = "search_res_400.json"
    os.makedirs("D:/disk/data/demo", exist_ok=True)
    os.makedirs("D:/disk/data/demo/video", exist_ok=True)
    i = 1
    for id in get_video_id(json_file):
        video_save_path = "D:/disk/data/demo/video"
        # down_cmd = f"yt-dlp -S ext https://www.youtube.com/watch?v={id} --output {video_save_path}/%(id)s.%(ext)s "
        # os.system(down_cmd)
        yt = YouTube(f'https://www.youtube.com/watch?v={id}')
        yt.streams.get_highest_resolution().download(video_save_path, filename=f"{i}_{id}.mp4")
        i = i + 1
        # shutil.move(osp.join(video_save_path, os.listdir(video_save_path)[0]), osp.join(video_save_path, f"{id}.mp4"))


if __name__ == '__main__':
    main()
