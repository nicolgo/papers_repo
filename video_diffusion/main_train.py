import torch
import torchvision.io

from vdm.models.imagen.elucidated_imagen import ElucidatedImagen
from vdm.models.imagen.trainer import ImagenTrainer
from vdm.models.imagen_video.imagen_video import Unet3D

if __name__ == "__main__":
    unet1 = Unet3D(dim=64, dim_mults=(1, 2, 4, 8)).cuda()

    unet2 = Unet3D(dim=64, dim_mults=(1, 2, 4, 8)).cuda()

    # elucidated imagen, which contains the unets above (base unet and super resoluting ones)

    imagen = ElucidatedImagen(unets=(unet1, unet2), image_sizes=(16, 32), random_crop_sizes=(None, 16),
                              num_sample_steps=10, cond_drop_prob=0.1, sigma_min=0.002,  # min noise level
                              sigma_max=(80, 160),  # max noise level, double the max noise level for upsampler
                              sigma_data=0.5,  # standard deviation of data distribution
                              rho=7,  # controls the sampling schedule
                              P_mean=-1.2,  # mean of log-normal distribution from which noise is drawn for training
                              P_std=1.2,
                              # standard deviation of log-normal distribution from which noise is drawn for training
                              S_churn=80,  # parameters for stochastic sampling - depends on dataset, Table 5 in apper
                              S_tmin=0.05, S_tmax=50, S_noise=1.003, ).cuda()

    # mock videos (get a lot of this) and text encodings from large T5

    texts = ['a whale breaching from afar', 'young girl blowing out candles on her birthday cake',
             'fireworks with blue and green sparkles', 'dust motes swirling in the morning sunshine on the windowsill']

    videos = torch.randn(4, 3, 10, 32, 32).cuda()  # (batch, channels, time / video frames, height, width)

    # feed images into imagen, training each unet in the cascade
    # for this example, only training unet 1

    trainer = ImagenTrainer(imagen)
    trainer(videos, texts=texts, unet_number=1)
    trainer.update(unet_number=1)

    videos = trainer.sample(texts=texts, video_frames=20)  # extrapolating to 20 frames from training on 10 frames

    videos.shape  # (4, 3, 20, 32, 32)
    # torchvision.io.write_video("test.mp4",
    #                            videos.view(videos.shape[0], videos.shape[-2], videos.shape[-1], videos.shape[1]), 10)
