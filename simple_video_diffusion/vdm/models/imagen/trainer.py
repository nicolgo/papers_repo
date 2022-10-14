import torch
from torch import nn
from math import ceil
from collections.abc import Iterable
from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs
from functools import partial
from torch.optim import Adam
from ema_pytorch import EMA
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import pytorch_warmup as warmup


def exists(val):
    return val is not None


def split_iterable(it, split_size):
    accum = []
    for ind in range(ceil(len(it) / split_size)):
        start_index = ind * split_size
        accum.append(it[start_index: (start_index + split_size)])
    return accum


def split(t, split_size=None):
    if not exists(split_size):
        return t

    if isinstance(t, torch.Tensor):
        return t.split(split_size, dim=0)

    if isinstance(t, Iterable):
        return split_iterable(t, split_size)

    return TypeError


def find_first(cond, arr):
    for el in arr:
        if cond(el):
            return el
    return None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def split_args_and_kwargs(*args, split_size=None, **kwargs):
    all_args = (*args, *kwargs.values())
    len_all_args = len(all_args)
    first_tensor = find_first(lambda t: isinstance(t, torch.Tensor), all_args)

    batch_size = len(first_tensor)
    split_size = default(split_size, batch_size)
    num_chunks = ceil(batch_size / split_size)

    dict_len = len(kwargs)
    dict_keys = kwargs.keys()
    split_kwargs_index = len_all_args - dict_len

    split_all_args = [split(arg, split_size=split_size) if exists(arg) and isinstance(arg, (torch.Tensor, Iterable))
                      else ((arg,) * num_chunks) for arg in all_args]
    chunk_sizes = tuple(map(len, split_all_args[0]))

    for (chunk_size, *chunked_all_args) in tuple(zip(chunk_sizes, *split_all_args)):
        chunked_args, chunked_kwargs_values = chunked_all_args[:split_kwargs_index], chunked_all_args[
                                                                                     split_kwargs_index:]
        chunked_kwargs = dict(tuple(zip(dict_keys, chunked_kwargs_values)))
        chunk_size_frac = chunk_size / batch_size
        yield chunk_size_frac, (chunked_args, chunked_kwargs)


def group_dict_by_key(cond, d):
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)


def string_begins_with(prefix, str):
    return str.startswith(prefix)


def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(
        map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs


def cast_tuple(val, length=1):
    if isinstance(val, list):
        val = tuple(val)

    return val if isinstance(val, tuple) else ((val,) * length)


class ImagenTrainer(nn.Module):
    def __init__(self, imagen=None, imagen_checkpoint_path=None, use_ema=True, lr=1e-4, eps=1e-8, beta1=0.9, beta2=0.99,
                 max_grad_norm=None, group_wd_params=True, warmup_steps=None, cosine_decay_max_steps=None,
                 only_train_unet_number=None, fp16=False, precision=None, split_batches=True,
                 dl_tuple_output_keywords_names=('images', 'text_embeds', 'text_masks', 'cond_images'), verbose=True,
                 split_valid_fraction=0.025, split_valid_from_train=False, split_random_seed=42, checkpoint_path=None,
                 checkpoint_every=None, checkpoint_fs=None, fs_kwargs: dict = None, max_checkpoints_keep=20, **kwargs):
        super().__init__()
        ema_kwargs, kwargs = groupby_prefix_and_trim('ema_', kwargs)
        self.is_elucidated = True

        self.imagen = imagen
        self.num_unets = len(self.imagen.unets)

        self.only_train_unet_number = only_train_unet_number
        self.validate_and_set_unet_being_trained(only_train_unet_number)

        accelerate_kwargs, kwargs = groupby_prefix_and_trim('accelerate_', kwargs)
        accelerator_mixed_precision = default(precision, 'fp16' if fp16 else 'no')
        self.accelerator = Accelerator(**{
            'split_batches': split_batches,
            'mixed_precision': accelerator_mixed_precision,
            'kwargs_handlers': [DistributedDataParallelKwargs(find_unused_parameters=True)], **accelerate_kwargs})

        self.use_ema = use_ema and self.is_main
        self.ema_unets = nn.ModuleList([])
        grad_scaler_enabled = fp16
        lr, eps, warmup_steps, cosine_decay_max_steps = map(partial(cast_tuple, length=self.num_unets),
                                                            (lr, eps, warmup_steps, cosine_decay_max_steps))
        for ind, (unet, unet_lr, unet_eps, unet_warmup_steps, unet_cosine_decay_max_steps) in enumerate(
                zip(self.imagen.unets, lr, eps, warmup_steps, cosine_decay_max_steps)):
            optimizer = Adam(unet.parameters(), lr=unet_lr, eps=unet_eps, betas=(beta1, beta2), **kwargs)
            if self.use_ema:
                self.ema_unets.append(EMA(unet, **ema_kwargs))

            scaler = GradScaler(enabled=grad_scaler_enabled)

            scheduler = warmup_scheduler = None
            if exists(unet_cosine_decay_max_steps):
                scheduler = CosineAnnealingLR(optimizer, T_max=unet_cosine_decay_max_steps)
            if exists(unet_warmup_steps):
                warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=unet_warmup_steps)
                if not exists(scheduler):
                    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

            setattr(self, f'optim{ind}', optimizer)
            setattr(self, f'scaler{ind}', scaler)
            setattr(self, f'scheduler{ind}', scheduler)
            setattr(self, f'warmup{ind}', warmup_scheduler)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def wrap_unet(self, unet_number):
        if hasattr(self, 'one_unet_wrapped'):
            return

        unet = self.imagen.get_unet(unet_number)
        self.unet_being_trained = self.accelerator.prepare(unet)
        unet_index = unet_number - 1

        optimizer = getattr(self, f'optim{unet_index}')
        scheduler = getattr(self, f'scheduler{unet_index}')
        optimizer = self.accelerator.prepare(optimizer)
        if exists(scheduler):
            scheduler = self.accelerator.prepare(scheduler)

        setattr(self, f'optim{unet_index}', optimizer)
        setattr(self, f'scheduler{unet_index}', scheduler)
        self.one_unet_wrapped = True

    def validate_and_set_unet_being_trained(self, unet_number=None):
        assert not exists(self.only_train_unet_number) or self.only_train_unet_number == unet_number, \
            'you cannot only train on one unet at a time. ' \
            'you will need to save the trainer into a checkpoint, and resume training on a new unet'
        self.only_train_unet_number = unet_number
        self.imagen.only_train_unet_number = unet_number
        if not exists(unet_number):
            return

        self.wrap_unet(unet_number)

    def set_accelerator_scaler(self, unet_number):
        scaler = getattr(self, f'scaler{unet_number - 1}')

        self.accelerator.scaler = scaler
        for optimizer in self.accelerator._optimizers:
            optimizer.scaler = scaler

    def forward(self, *args, unet_number=None, max_batch_size=None, **kwargs):
        unet_number = unet_number if unet_number is not None else self.num_unets
        self.validate_and_set_unet_being_trained(unet_number)
        self.set_accelerator_scaler(unet_number)

        total_loss = 0.
        for chunk_size_frac, (chunked_args, chunked_kwargs) in split_args_and_kwargs(*args, split_size=max_batch_size,
                                                                                     **kwargs):
            with self.accelerator.autocast():
                loss = self.imagen(*chunked_args, unet=self.unet_being_trained, unet_number=unet_number,
                                   **chunked_kwargs)
                loss = loss * chunk_size_frac

            total_loss += loss.item()

            if self.training:
                self.accelerator.backward(loss)

        return total_loss
