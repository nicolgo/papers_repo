import torch
from collections import namedtuple
from torch import nn
from .utils import *
from torch.nn.parallel import DistributedDataParallel
from vdm.models.imagen_video.imagen_video import (Unet3D, resize_video_to)
from vdm.models.imagen.t5 import t5_encode_text, get_encoded_dim, DEFAULT_T5_NAME
from functools import partial
from einops import rearrange, repeat, reduce
from einops_exts import rearrange_many, repeat_many, check_shape
from torch.cuda.amp import autocast
import kornia.augmentation as K
from random import random

from vdm.models.imagen.variational_diffusion import GaussianDiffusionContinuousTimes

Hparams_fields = ['num_sample_steps', 'sigma_min', 'sigma_max', 'sigma_data', 'rho', 'P_mean', 'P_std', 'S_churn',
                  'S_tmin', 'S_tmax', 'S_noise']
Hparams = namedtuple('Hparams', Hparams_fields)


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


class ElucidatedImagen(nn.Module):
    def __init__(self, unets, *, image_sizes,  # for cascading ddpm, image size at each stage
                 text_encoder_name=DEFAULT_T5_NAME, text_embed_dim=None, channels=3, cond_drop_prob=0.1,
                 random_crop_sizes=None, lowres_sample_noise_level=0.2,
                 # in the paper, they present a new trick where they noise the lowres conditioning image, and at sample time, fix it to a certain level (0.1 or 0.3) - the unets are also made to be conditioned on this noise level
                 per_sample_random_aug_noise_level=False,
                 # unclear when conditioning on augmentation noise level, whether each batch element receives a random aug noise value - turning off due to @marunine's find
                 condition_on_text=True, auto_normalize_img=True,
                 # whether to take care of normalizing the image from [0, 1] to [-1, 1] and back automatically - you can turn this off if you want to pass in the [-1, 1] ranged image yourself from the dataloader
                 dynamic_thresholding=True, dynamic_thresholding_percentile=0.95,
                 # unsure what this was based on perusal of paper
                 only_train_unet_number=None, lowres_noise_schedule='linear', num_sample_steps=32,
                 # number of sampling steps
                 sigma_min=0.002,  # min noise level
                 sigma_max=80,  # max noise level
                 sigma_data=0.5,  # standard deviation of data distribution
                 rho=7,  # controls the sampling schedule
                 P_mean=-1.2,  # mean of log-normal distribution from which noise is drawn for training
                 P_std=1.2,  # standard deviation of log-normal distribution from which noise is drawn for training
                 S_churn=80,  # parameters for stochastic sampling - depends on dataset, Table 5 in apper
                 S_tmin=0.05, S_tmax=50, S_noise=1.003, ):
        super().__init__()

        self.only_train_unet_number = only_train_unet_number

        # conditioning hparams
        self.condition_on_text = condition_on_text
        self.unconditional = not condition_on_text

        # channels
        self.channels = channels

        # automatically take care of ensuring that first unet is unconditional
        # while the rest of the unets are conditioned on the low resolution image produced by previous unet
        unets = cast_tuple(unets)
        num_unets = len(unets)

        # randomly cropping for upsampler training
        self.random_crop_sizes = cast_tuple(random_crop_sizes, num_unets)

        # lowres augmentation noise schedule
        self.lowres_noise_schedule = GaussianDiffusionContinuousTimes(noise_schedule=lowres_noise_schedule)

        # get text encoder
        self.text_encoder_name = text_encoder_name
        self.text_embed_dim = default(text_embed_dim, lambda: get_encoded_dim(text_encoder_name))
        self.encode_text = partial(t5_encode_text, name=text_encoder_name)

        # construct unets
        self.unets = nn.ModuleList([])
        # keeps track of which unet is being trained at the moment
        self.unet_being_trained_index = -1

        for ind, one_unet in enumerate(unets):
            is_first = ind == 0
            one_unet = one_unet.cast_model_parameters(lowres_cond=not is_first, cond_on_text=self.condition_on_text,
                                                      text_embed_dim=self.text_embed_dim if self.condition_on_text else None,
                                                      channels=self.channels, channels_out=self.channels)
            self.unets.append(one_unet)

        # determine whether we are training on images or video
        is_video = any([isinstance(unet, Unet3D) for unet in self.unets])
        self.is_video = is_video
        self.right_pad_dims_to_datatype = partial(rearrange,
                                                  pattern=('b -> b 1 1 1' if not is_video else 'b -> b 1 1 1 1'))
        self.resize_to = resize_video_to if is_video else resize_image_to

        # unet image sizes
        self.image_sizes = cast_tuple(image_sizes)
        self.sample_channels = cast_tuple(self.channels, num_unets)

        # cascading ddpm related stuff
        lowres_conditions = tuple(map(lambda t: t.lowres_cond, self.unets))
        self.lowres_sample_noise_level = lowres_sample_noise_level
        self.per_sample_random_aug_noise_level = per_sample_random_aug_noise_level

        # classifier free guidance
        self.cond_drop_prob = cond_drop_prob
        self.can_classifier_guidance = cond_drop_prob > 0.

        # normalize and unnormalize image functions
        self.normalize_img = normalize_neg_one_to_one if auto_normalize_img else identity
        self.unnormalize_img = unnormalize_zero_to_one if auto_normalize_img else identity
        self.input_image_range = (0. if auto_normalize_img else -1., 1.)

        # dynamic thresholding
        self.dynamic_thresholding = cast_tuple(dynamic_thresholding, num_unets)
        self.dynamic_thresholding_percentile = dynamic_thresholding_percentile

        # elucidating parameters
        hparams = [num_sample_steps, sigma_min, sigma_max, sigma_data, rho,
                   P_mean, P_std, S_churn, S_tmin, S_tmax, S_noise, ]
        hparams = [cast_tuple(hp, num_unets) for hp in hparams]
        self.hparams = [Hparams(*unet_hp) for unet_hp in zip(*hparams)]

        # one temp parameter for keeping track of device
        self.register_buffer('_temp', torch.tensor([0.]), persistent=False)

        # default to device of unets passed in
        self.to(next(self.unets.parameters()).device)

    def c_skip(self, sigma_data, sigma):
        return (sigma_data ** 2) / (sigma ** 2 + sigma_data ** 2)

    def c_in(self, sigma_data, sigma):
        return 1 * (sigma ** 2 + sigma_data ** 2) ** -0.5

    def c_out(self, sigma_data, sigma):
        return sigma * sigma_data * (sigma_data ** 2 + sigma ** 2) ** -0.5

    def c_noise(self, sigma):
        return log(sigma) * 0.25

    # preconditioned network output equation (7) in the paper
    def preconditioned_network_forward(self, unet_forward, noised_images, sigma, *, sigma_data, clamp=False,
                                       dynamic_threshold=True, **kwargs):
        batch, device = noised_images.shape[0], noised_images.device
        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device=device)
        padded_sigma = self.right_pad_dims_to_datatype(sigma)
        net_out = unet_forward(self.c_in(sigma_data, padded_sigma) * noised_images, self.c_noise(sigma), **kwargs)
        out = self.c_skip(sigma_data, padded_sigma) * noised_images + self.c_out(sigma_data, padded_sigma) * net_out
        if not clamp:
            return out
        return self.threshold_x_start(out, dynamic_threshold)

    @property
    def device(self):
        return self._temp.device

    def get_unet(self, unet_number):
        assert 0 < unet_number <= len(self.unets)
        index = unet_number - 1

        if isinstance(self.unets, nn.ModuleList):
            unets_list = [unet for unet in self.unets]
            delattr(self, 'unets')
            self.unets = unets_list

        if index != self.unet_being_trained_index:
            for unet_index, unet in enumerate(self.unets):
                unet.to(self.device if unet_index == index else 'cpu')

        self.unet_being_trained_index = index
        return self.unets[index]

    def loss_weight(self, sigma_data, sigma):
        return (sigma ** 2 + sigma_data ** 2) * (sigma * sigma_data) ** -2

    def noise_distribution(self, P_mean, P_std, batch_size):
        return (P_mean + P_std * torch.randn((batch_size,), device=self.device)).exp()

    def forward(self, images, unet=None, texts=None, text_embeds=None, text_masks=None, unet_number=None,
                cond_images=None):
        unet_number = default(unet_number, 1)
        images = cast_uint8_images_to_float(images)
        cond_images = cond_images if cond_images is None else cast_uint8_images_to_float(cond_images)

        unet_index = unet_number - 1
        unet = default(unet, lambda: self.get_unet(unet_number))

        target_image_size = self.image_sizes[unet_index]
        random_crop_size = self.random_crop_sizes[unet_index]
        prev_image_size = self.image_sizes[unet_index - 1] if unet_index > 0 else None
        hp = self.hparams[unet_index]

        batch_size, c, *_, h, w, device, is_video = *images.shape, images.device, (images.ndim == 5)
        frames = images.shape[2] if is_video else None
        check_shape(images, 'b c ...', c=self.channels)

        if exists(texts) and not exists(text_embeds) and not self.unconditional:
            with autocast(enabled=False):
                text_embeds, text_masks = self.encode_text(texts, return_attn_mask=True)
            text_embeds, text_masks = map(lambda t: t.to(images.device), (text_embeds, text_masks))

        if not self.unconditional:
            text_masks = default(text_masks, lambda: torch.any(text_embeds != 0., dim=-1))
        lowres_cond_img = lowres_aug_times = None

        if exists(prev_image_size):
            lowres_cond_img = self.resize_to(images, prev_image_size, clamp_range=self.input_image_range)
            lowres_cond_img = self.resize_to(lowres_cond_img, target_image_size, clamp_range=self.input_image_range)
            if self.per_sample_random_aug_noise_level:
                lowres_aug_times = self.lowres_noise_schedule.sample_random_times(batch_size, device=device)
            else:
                lowres_aug_time = self.lowres_noise_schedule.sample_random_times(1, device=device)
                lowres_aug_times = repeat(lowres_aug_time, '1 -> b', b=batch_size)

        images = self.resize_to(images, target_image_size)

        # normalize to [-1, 1]
        images = self.normalize_img(images)

        lowres_cond_img = maybe(self.normalize_img)(lowres_cond_img)
        # random cropping during training for upsamplers
        if exists(random_crop_size):
            aug = K.RandomCrop((random_crop_size, random_crop_size), p=1.)
            if is_video:
                images, lowres_cond_img = rearrange_many((images, lowres_cond_img), 'b c f h w -> (b f) c h w')
            images = aug(images)
            lowres_cond_img = aug(lowres_cond_img, params=aug._params)
            if is_video:
                images, lowres_cond_img = rearrange_many((images, lowres_cond_img), '(b f) c h w -> b c f h w',
                                                         f=frames)

        lowres_cond_img_noisy = None
        if exists(lowres_cond_img):
            lowres_cond_img_noisy, _ = self.lowres_noise_schedule.q_sample(x_start=lowres_cond_img, t=lowres_aug_times,
                                                                           noise=torch.randn_like(lowres_cond_img))

        sigmas = self.noise_distribution(hp.P_mean, hp.P_std, batch_size)
        padded_sigmas = self.right_pad_dims_to_datatype(sigmas)
        # noise
        noise = torch.randn_like(images)
        noised_images = images + padded_sigmas * noise
        # unet kwargs
        unet_kwargs = dict(sigma_data=hp.sigma_data, text_embeds=text_embeds, text_mask=text_masks,
                           cond_images=cond_images,
                           lowres_noise_times=self.lowres_noise_schedule.get_condition(lowres_aug_times),
                           lowres_cond_img=lowres_cond_img_noisy, cond_drop_prob=self.cond_drop_prob, )
        # self conditioning - https://arxiv.org/abs/2208.04202
        self_cond = unet.module.self_cond if isinstance(unet, DistributedDataParallel) else unet
        if self_cond and random() < 0.5:
            with torch.no_grad():
                pred_x0 = self.preconditioned_network_forward(unet.forward, noised_images, sigmas,
                                                              **unet_kwargs).detach()
            unet_kwargs = {**unet_kwargs, 'self_cond': pred_x0}

        # get prediction
        denoised_images = self.preconditioned_network_forward(unet.forward, noised_images, sigmas, **unet_kwargs)
        # losses
        losses = F.mse_loss(denoised_images, images, reduction='none')
        losses = reduce(losses, 'b ... -> b', 'mean')
        losses = losses * self.loss_weight(hp.sigma_data, sigmas)
        # return average loss
        return losses.mean()
