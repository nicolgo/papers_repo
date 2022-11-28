import os

import gdown
import numpy as np
import torch
from tqdm import tqdm
from datasets.video_dataset import VideoDataset
from vdm.modules.fvd.fvd import get_fvd_logits, frechet_distance
from vdm.models.video_diffusion.video_diffusion import num_to_groups
from vdm.models.video_diffusion import Unet3D, GaussianDiffusion
from vdm.models.video_diffusion.video_trainer import VideoTrainer


def download(id, fname, root=os.path.expanduser('~/.cache/videogpt')):
    os.makedirs(root, exist_ok=True)
    destination = os.path.join(root, fname)

    if os.path.exists(destination):
        return destination

    gdown.download(id=id, output=destination, quiet=False)
    return destination


_I3D_PRETRAINED_ID = '1mQK8KD8G6UWRa5t87SRMm5PVXtlpneJT'


def load_i3d_pretrained(device=torch.device('cpu')):
    from vdm.modules.fvd.pytorch_i3d import InceptionI3d
    i3d = InceptionI3d(400, in_channels=3).to(device)
    filepath = download(_I3D_PRETRAINED_ID, 'i3d_pretrained_400.pt')
    i3d.load_state_dict(torch.load(filepath, map_location=device))
    i3d.eval()
    return i3d


def main():
    device = torch.device(f"cuda:{0}")
    all_batch_size = 16
    sample_iters = 25
    sample_milestone = 215
    model = Unet3D(dim=64, cond_dim=64, dim_mults=(1, 2, 4, 8), )
    diffusion = GaussianDiffusion(model, image_size=64, num_frames=10, timesteps=1000, loss_type='l2').cuda()

    trainer = VideoTrainer(diffusion, "D:/ssd/nicol/papers_repo/simple_video_diffusion/data/ucf101",
                           train_batch_size=all_batch_size,
                           train_lr=1e-4, save_and_sample_every=2000, train_num_steps=700000,  # total training steps
                           gradient_accumulate_every=2,  # gradient accumulation steps
                           ema_decay=0.995,  # exponential moving average decay
                           amp=True,  # turn on mixed precision
                           num_sample_rows=1)
    trainer.load(milestone=sample_milestone)

    # for i in range(100):
    #     num_samples = 1
    #     batches = num_to_groups(num_samples, trainer.batch_size)
    #     all_videos_list = list(
    #         map(lambda n: trainer.ema_model.sample(batch_size=n, cond=torch.tensor([0, 1, 2, 3, 4]).cuda()), batches))
    #     all_videos_list = torch.cat(all_videos_list, dim=0)
    #     all_videos_list = torch.nn.functional.pad(all_videos_list, (2, 2, 2, 2))
    #     one_gif = rearrange(all_videos_list, '(i j) c f h w -> c f (i h) (j w)', i=trainer.num_sample_rows)
    #     video_path = str(trainer.results_folder / str(f'sample_{i}_{sample_milestone}.gif'))
    #     video_tensor_to_gif(one_gif, video_path)

    folder = "D:/ssd/nicol/papers_repo/simple_video_diffusion/data/dataset_split"
    test_dataset = VideoDataset(folder, sequence_length=10, train=False, resolution=64)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=all_batch_size, num_workers=0,
                                                  pin_memory=True,
                                                  shuffle=False)

    #################### Load I3D ########################################
    i3d = load_i3d_pretrained(device)

    #################### Compute FVD ###############################
    fvds = []
    fvds_star = []
    for _ in tqdm(range(sample_iters)):
        fvd = eval_fvd(i3d, trainer, test_dataloader, device)
        fvds.append(fvd)
        # fvds_star.append(fvd_star)

    fvd_mean = np.mean(fvds)
    fvd_std = np.std(fvds)

    # fvd_star_mean = np.mean(fvds_star)
    # fvd_star_std = np.std(fvds_star)

    print(f"Final FVD {fvd_mean:.2f} +/- {fvd_std:.2f}")


def eval_fvd(i3d, trainer, loader, device):
    batch = next(iter(loader))
    batch = {k: v.to(device) for k, v in batch.items()}

    real = batch['video'].to(device)
    real = real + 0.5
    real = real.permute(0, 2, 3, 4, 1).cpu().numpy()  # BCTHW -> BTHWC
    real = (real * 255).astype('uint8')
    real_embeddings = get_fvd_logits(real, i3d=i3d, device=device)

    fake_embeddings = []
    # for i in range(0, batch['video'].shape[0]):
    num_samples = 1
    batches = num_to_groups(num_samples, trainer.batch_size)
    all_videos_list = list(
        map(lambda n: trainer.ema_model.sample(batch_size=n, cond=(batch['label'].to(device))), batches))
    fake = torch.cat(all_videos_list, dim=0)
    fake = (fake + 0.5).clamp(0, 1)
    fake = fake.permute(0, 2, 3, 4, 1).cpu().numpy()  # BCTHW -> BTHWC
    fake = (fake * 255).astype('uint8')
    fake_embeddings.append(get_fvd_logits(fake, i3d=i3d, device=device))
    fake_embeddings = torch.cat(fake_embeddings)

    # assert fake_embeddings.shape[0] == real_embeddings.shape[0] == 256

    fvd = frechet_distance(fake_embeddings.clone(), real_embeddings)

    return fvd.item()


if __name__ == '__main__':
    main()
