import torch
from collections import namedtuple
from torch import nn
from .utils import *
from typing import List, Union
from torch.nn.parallel import DistributedDataParallel
from vdm.models.imagen_video.imagen_video import (Unet3D, resize_video_to)
from vdm.models.imagen.t5 import t5_encode_text, get_encoded_dim, DEFAULT_T5_NAME
from functools import partial
from einops import rearrange

from vdm.models.imagen.variational_diffusion import GaussianDiffusionContinuousTimes

Hparams_fields = ['num_sample_steps', 'sigma_min', 'sigma_max', 'sigma_data', 'rho', 'P_mean', 'P_std', 'S_churn',
                  'S_tmin', 'S_tmax', 'S_noise']
Hparams = namedtuple('Hparams', Hparams_fields)


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
        assert num_unets == len(self.image_sizes), f'you did not supply the correct ' \
                                                   f'number of u-nets ({len(self.unets)}) for resolutions {image_sizes}'
        self.sample_channels = cast_tuple(self.channels, num_unets)

        # cascading ddpm related stuff
        lowres_conditions = tuple(map(lambda t: t.lowres_cond, self.unets))
        assert lowres_conditions == (False, *((True,) * (num_unets - 1))), \
            'the first unet must be unconditioned (by low resolution image), and the rest of the unets must have `lowres_cond` set to True'

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

    def forward(self, images, unet: Union[Unet3D, DistributedDataParallel] = None,
                unet_number=None, cond_images=None):
        unet_number = default(unet_number, 1)
        images = cast_uint8_images_to_float(images)
        cond_images = cond_images if cond_images is not None else cast_uint8_images_to_float(cond_images)

        unet_index = unet_number - 1
        unet = default(unet, lambda: self.get_unet(unet_number))

        target_image_size = self.image_sizes[unet_index]
        random_crop_size = self.random_crop_sizes[unet_index]
        prev_image_size = self.image_sizes[unet_index - 1] if unet_index > 0 else None
        hp = self.hparams[unet_index]
