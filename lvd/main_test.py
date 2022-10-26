# check vqvae
import torch
from torchvision.io import read_video, read_video_timestamps
from lvd.models.vqvae import VQVAE
from datasets.video_dataset import preprocess
from matplotlib import pyplot as plt
from matplotlib import animation


def test_vq_vae(checkpoint_file, video_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vq_vae = VQVAE.load_from_checkpoint(checkpoint_file).to(device)
    vq_vae.eval()
    resolution, sequence_length = vq_vae.args.resolution, 16

    pts = read_video_timestamps(video_name, pts_unit="sec")[0]
    orig_video = read_video(video_name, pts_unit='sec', start_pts=pts[0], end_pts=pts[sequence_length - 1])[0]
    orig_video = preprocess(orig_video, resolution, sequence_length).unsqueeze(0).to(device)

    with torch.no_grad():
        encodings = vq_vae.encode(orig_video)
        video_recon = vq_vae.decode(encodings)
        video_recon = torch.clamp(video_recon, -0.5, 0.5)
    return orig_video, video_recon


def show_two_vides(video, video_recon):
    # draw videos
    videos = torch.cat((video, video_recon), dim=-1)
    videos = videos[0].permute(1, 2, 3, 0)  # CTHW -> THWC
    videos = ((videos + 0.5) * 255).cpu().numpy().astype('uint8')

    fig = plt.figure()
    plt.title('real (left), reconstruction (right)')
    plt.axis('off')
    im = plt.imshow(videos[0, :, :, :])
    plt.close()

    def init():
        im.set_data(videos[0, :, :, :])

    def animate(i):
        im.set_data(videos[i, :, :, :])
        return im

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=videos.shape[0], interval=200)
    anim.save("compare.gif")


if __name__ == "__main__":
    file_path = "pretrained/epoch=35-step=8676.ckpt"
    video_filename = "data/ucf101/test/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi"
    video, video_recon = test_vq_vae(file_path, video_filename)
    show_two_vides(video, video_recon)
