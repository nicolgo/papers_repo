import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '5,6'

import torch
import torch.nn as nn

from vdm.models.video_diffusion import Unet3D, GaussianDiffusion
from vdm.models.video_diffusion.video_trainer import VideoTrainer

# from utils.vision_util import save_as_gif

if __name__ == "__main__":
    print(f"The model running on {torch.cuda.device_count()} GPUS")

    model = Unet3D(dim=64, cond_dim=64, dim_mults=(1, 2, 4, 8), )

    if torch.cuda.device_count() > 1:
        diffusion = nn.DataParallel(
            GaussianDiffusion(model, image_size=64, num_frames=10, timesteps=1000, loss_type='l1'))
        diffusion = diffusion.cuda()
    else:
        diffusion = GaussianDiffusion(model, image_size=64, num_frames=12, timesteps=500, loss_type='l1').cuda()

    trainer = VideoTrainer(diffusion, "D:/ssd/nicol/papers_repo/simple_video_diffusion/data/dataset_split", train_batch_size=2,
                           train_lr=1e-4, save_and_sample_every=2000, train_num_steps=700000,  # total training steps
                           gradient_accumulate_every=2,  # gradient accumulation steps
                           ema_decay=0.995,  # exponential moving average decay
                           amp=True,  # turn on mixed precision
                           num_sample_rows=1)

    trainer.train(prob_focus_present=0.5)
