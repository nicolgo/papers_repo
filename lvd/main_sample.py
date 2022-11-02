import torch
from omegaconf import OmegaConf
from lvd.util import instantiate_from_config
from lvd.models.diffusion.ddim import DDIMSampler
from utils.vision_util import save_as_gif


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location='cuda:0')
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model():
    config = OmegaConf.load("configs/linux_lvd_3d.yaml")
    model = load_model_from_config(config, "pretrained/epoch=3-step=3844.ckpt")
    return model


model = get_model()
sampler = DDIMSampler(model)

classes = [0]  # define classes to be sampled here
n_samples_per_class = 6

ddim_steps = 20
ddim_eta = 0.0
scale = 3.0  # for unconditional guidance

all_samples = list()
CUDA_LAUNCH_BLOCKING = 1
with torch.no_grad():
    with model.ema_scope():
        # uc = model.get_learned_conditioning(
        #     {model.cond_stage_key: torch.tensor(n_samples_per_class * [1000]).to(model.device)})

        for class_label in classes:
            print(
                f"rendering {n_samples_per_class} examples of class '{class_label}' in {ddim_steps} steps and using s={scale:.2f}.")
            xc = torch.tensor(n_samples_per_class * [class_label]).to(model.device)
            c = model.get_learned_conditioning({model.cond_stage_key: xc})

            samples_ddim, _ = sampler.sample(S=ddim_steps, conditioning=c, batch_size=n_samples_per_class,
                                             shape=[3, 16, 32, 32], verbose=False, unconditional_guidance_scale=scale,
                                             unconditional_conditioning=None, eta=ddim_eta)

            # x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            for i in range(0, len(x_samples_ddim)):
                save_as_gif(x_samples_ddim[i], f"sample{i}.gif")
            all_samples.append(x_samples_ddim)
