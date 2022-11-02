import time
import os
import torch
import torchvision.io
from vdm.models.imagen.elucidated_imagen import ElucidatedImagen
from vdm.models.imagen.trainer import ImagenTrainer
from vdm.models.imagen_video.imagen_video import Unet3D
from datasets.UCF101 import UCF101Wrapper
from root_dir import ROOT_DIR
# from utils.vision_util import save_as_gif
import math

if __name__ == "__main__":
    unet1 = Unet3D(dim=64, dim_mults=(1, 2, 4, 8)).cuda()

    unet2 = Unet3D(dim=64, dim_mults=(1, 2, 4, 8)).cuda()

    # elucidated imagen, which contains the unets above (base unet and super resoluting ones)
    imagen = ElucidatedImagen(unets=(unet1, unet2), image_sizes=(64, 128), random_crop_sizes=(None, 64),
                              condition_on_text=False, num_sample_steps=10, cond_drop_prob=0.1,
                              sigma_min=0.002,  # min noise level
                              sigma_max=(80, 160),  # max noise level, double the max noise level for upsampler
                              sigma_data=0.5,  # standard deviation of data distribution
                              rho=7,  # controls the sampling schedule
                              P_mean=-1.2,  # mean of log-normal distribution from which noise is drawn for training
                              P_std=1.2,  # standard deviation of log-normal distribution
                              S_churn=80,  # parameters for stochastic sampling - depends on dataset, Table 5 in apper
                              S_tmin=0.05, S_tmax=50, S_noise=1.003, ).cuda()
    data_path = "D:/ssd/nicol/papers_repo/simple_video_diffusion/data/UCF-101"
    dataset = UCF101Wrapper(data_path, False, 128, data_path, xflip=False, return_vid=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    # video = next(iter(dataloader))
    # mock videos (get a lot of this) and text encodings from large T5
    # texts = ['a whale breaching from afar', 'young girl blowing out candles on her birthday cake',
    #          'fireworks with blue and green sparkles', 'dust motes swirling in the morning sunshine on the windowsill']
    i = 0
    logdir = os.path.join('./logs', time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime()))
    out_dir = ROOT_DIR + "/outputs"
    os.makedirs(out_dir, exist_ok=True)
    try:
        while i < 10000:
            # (batch, channels, time / video frames, height, width)
            videos = next(iter(dataloader))
            videos = videos.cuda()
            # videos = torch.randn(4, 3, 10, 32, 32).cuda()
            # feed images into imagen, training each unet in the cascade, for this example, only training unet 1
            trainer = ImagenTrainer(imagen, checkpoint_path=logdir, checkpoint_every=1)
            trainer(videos, unet_number=1)
            trainer.update(unet_number=1)
            i += 1
            if i % 100 == 0:
                videos = trainer.sample(video_frames=20)  # extrapolating to 20 frames from training on 10 frames
                # save_as_gif(videos[0], os.path.join(out_dir, f'generate_videos{i}.gif'))
    except KeyboardInterrupt:
        print("Terminating...")
