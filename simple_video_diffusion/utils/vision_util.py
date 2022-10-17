import numpy as np
import torch
import torchvision
import PIL
import matplotlib.pyplot as plt
from einops import rearrange
import math
import numpy as np
import skvideo
import skvideo.io
import imageio


# deprecated
def save_image_grid(img, fname, drange, grid_size, normalize=True):
    if normalize:
        lo, hi = drange
        img = np.asarray(img, dtype=np.float32)
        img = (img - lo) * (255 / (hi - lo))
        img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, T, H, W = img.shape
    img = img.reshape(gh, gw, C, T, H, W)
    img = img.transpose(3, 0, 4, 1, 5, 2)
    img = img.reshape(T, gh * H, gw * W, C)

    print(f'Saving Video with {T} frames, img shape {H}, {W}')

    assert C in [3]

    if C == 3:
        torchvision.io.write_video(f'{fname[:-3]}mp4', torch.from_numpy(img), fps=8)
        imgs = [PIL.Image.fromarray(img[i], 'RGB') for i in range(len(img))]
        imgs[0].save(fname, quality=95, save_all=True, append_images=imgs[1:], duration=100, loop=0)


def save_video_grid(video, fname, nrow=None):
    b, c, t, h, w = video.shape
    video = video.permute(0, 2, 3, 4, 1)
    video = (video.cpu().numpy() * 255).astype('uint8')

    if nrow is None:
        nrow = math.ceil(math.sqrt(b))
    ncol = math.ceil(b / nrow)
    padding = 1
    video_grid = np.zeros((t, (padding + h) * nrow + padding, (padding + w) * ncol + padding, c), dtype='uint8')
    for i in range(b):
        r = i // ncol
        c = i % ncol

        start_r = (padding + h) * r
        start_c = (padding + w) * c
        video_grid[:, start_r:start_r + h, start_c:start_c + w] = video[i]
    skvideo.io.vwrite(fname, video_grid, inputdict={'-r': '5'})
    print('saved videos to', fname)


def show_grid_images(imgs):
    _, axs = plt.subplots(4, 4, figsize=(12, 12))
    axs = axs.flatten()
    i = 0
    for img, ax in zip(imgs, axs):
        image = rearrange(imgs[i], "c h w -> h w c")
        ax.imshow(image)
        i += 1
    plt.show()


def save_as_gif(images, file_name):
    images = images.permute(1, 2, 3, 0)
    image_list = []
    for i in range(images.shape[0]):
        image_list.append(images[i])
    imageio.mimwrite(file_name, image_list, fps=4)
