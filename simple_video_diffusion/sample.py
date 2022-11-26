import torch
from einops import rearrange

from vdm.models.video_diffusion import Unet3D, GaussianDiffusion
from vdm.models.video_diffusion.video_diffusion import num_to_groups, video_tensor_to_gif
from vdm.models.video_diffusion.video_trainer import VideoTrainer

# from utils.vision_util import save_as_gif

if __name__ == "__main__":
    print(f"The model running on {torch.cuda.device_count()} GPUS")
    sample_milestone = 53
    model = Unet3D(dim=64, cond_dim=64, dim_mults=(1, 2, 4, 8), )
    diffusion = GaussianDiffusion(model, image_size=64, num_frames=12, timesteps=500, loss_type='l2').cuda()

    trainer = VideoTrainer(diffusion, "D:/ssd/nicol/papers_repo/simple_video_diffusion/data/dataset_split", train_batch_size=1,
                           train_lr=1e-4, save_and_sample_every=2000, train_num_steps=700000,  # total training steps
                           gradient_accumulate_every=2,  # gradient accumulation steps
                           ema_decay=0.995,  # exponential moving average decay
                           amp=True,  # turn on mixed precision
                           num_sample_rows=1)
    trainer.load(milestone=sample_milestone)

    for i in range(10):
        num_samples = 1
        batches = num_to_groups(num_samples, trainer.batch_size)
        all_videos_list = list(
            map(lambda n: trainer.ema_model.sample(batch_size=n, cond=torch.tensor([0, 1, 2, 3, 4]).cuda()), batches))
        all_videos_list = torch.cat(all_videos_list, dim=0)
        all_videos_list = torch.nn.functional.pad(all_videos_list, (2, 2, 2, 2))
        one_gif = rearrange(all_videos_list, '(i j) c f h w -> c f (i h) (j w)', i=trainer.num_sample_rows)
        video_path = str(trainer.results_folder / str(f'sample_{i}_{sample_milestone}.gif'))
        video_tensor_to_gif(one_gif, video_path)
