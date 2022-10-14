import torch
from torch import nn
from math import ceil
from collections.abc import Iterable
from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs
from functools import partial, wraps
from torch.optim import Adam
from ema_pytorch import EMA
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import pytorch_warmup as warmup
from contextlib import contextmanager, nullcontext
import torch.nn.functional as F
import numpy as np
import os

from vdm.models.imagen.elucidated_imagen import NullUnet


def exists(val):
    return val is not None


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def cast_torch_tensor(fn, cast_fp16=False):
    @wraps(fn)
    def inner(model, *args, **kwargs):
        device = kwargs.pop('_device', model.device)
        cast_device = kwargs.pop('_cast_device', True)

        should_cast_fp16 = cast_fp16 and model.cast_half_at_training

        kwargs_keys = kwargs.keys()
        all_args = (*args, *kwargs.values())
        split_kwargs_index = len(all_args) - len(kwargs_keys)
        all_args = tuple(map(lambda t: torch.from_numpy(t) if exists(t) and isinstance(t, np.ndarray) else t, all_args))

        if cast_device:
            all_args = tuple(map(lambda t: t.to(device) if exists(t) and isinstance(t, torch.Tensor) else t, all_args))

        if should_cast_fp16:
            all_args = tuple(
                map(lambda t: t.half() if exists(t) and isinstance(t, torch.Tensor) and t.dtype != torch.bool else t,
                    all_args))

        args, kwargs_values = all_args[:split_kwargs_index], all_args[split_kwargs_index:]
        kwargs = dict(tuple(zip(kwargs_keys, kwargs_values)))

        out = fn(model, *args, **kwargs)
        return out

    return inner


def imagen_sample_in_chunks(fn):
    @wraps(fn)
    def inner(self, *args, max_batch_size=None, **kwargs):
        if not exists(max_batch_size):
            return fn(self, *args, **kwargs)

        if self.imagen.unconditional:
            batch_size = kwargs.get('batch_size')
            batch_sizes = num_to_groups(batch_size, max_batch_size)
            outputs = [fn(self, *args, **{**kwargs, 'batch_size': sub_batch_size}) for sub_batch_size in batch_sizes]
        else:
            outputs = [fn(self, *chunked_args, **chunked_kwargs) for _, (chunked_args, chunked_kwargs) in
                       split_args_and_kwargs(*args, split_size=max_batch_size, **kwargs)]

        if isinstance(outputs[0], torch.Tensor):
            return torch.cat(outputs, dim=0)

        return list(map(lambda t: torch.cat(t, dim=0), list(zip(*outputs))))

    return inner


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

        accelerate_kwargs, kwargs = groupby_prefix_and_trim('accelerate_', kwargs)
        accelerator_mixed_precision = default(precision, 'fp16' if fp16 else 'no')
        self.accelerator = Accelerator(**{
            'split_batches': split_batches,
            'mixed_precision': accelerator_mixed_precision,
            'kwargs_handlers': [DistributedDataParallelKwargs(find_unused_parameters=True)], **accelerate_kwargs})

        self.use_ema = use_ema and self.is_main  # used for update not forward
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

        self.only_train_unet_number = only_train_unet_number
        self.validate_and_set_unet_being_trained(only_train_unet_number)

        self.max_grad_norm = max_grad_norm
        self.ema_unet_being_trained_index = -1
        self.register_buffer('steps', torch.tensor([0] * self.num_unets))

        # check point
        self.checkpoint_path = checkpoint_path
        self.checkpoint_every = checkpoint_every  # save checkpoint duration
        self.max_checkpoints_keep = max_checkpoints_keep

        self.verbose = verbose

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

    @property
    def unets(self):
        return nn.ModuleList([ema.ema_model for ema in self.ema_unets])

    @property
    def device(self):
        return self.accelerator.device

    def get_ema_unet(self, unet_number=None):
        if not self.use_ema:
            return
        index = unet_number - 1
        if isinstance(self.unets, nn.ModuleList):
            unets_list = [unet for unet in self.ema_unets]
            delattr(self, 'ema_unets')
            self.ema_unets = unets_list
        if index != self.ema_unet_being_trained_index:
            for unet_index, unet in enumerate(self.ema_unets):
                unet.to(self.device if unet_index == index else 'cpu')
        self.ema_unet_being_trained_index = index
        return self.ema_unets[index]

    def save_to_checkpoint_folder(self):
        self.accelerator.wait_for_everyone()

        if not self.can_checkpoint:
            return

        total_steps = int(self.steps.sum().item())
        filepath = os.path.join(self.checkpoint_path, f'checkpoint.{total_steps}.pt')

        self.save(filepath)

        if self.max_checkpoints_keep <= 0:
            return

        sorted_checkpoints = self.all_checkpoints_sorted
        checkpoints_to_discard = sorted_checkpoints[self.max_checkpoints_keep:]

        for checkpoint in checkpoints_to_discard:
            self.fs.rm(checkpoint)

    def update(self, unet_number=None):
        unet_number = unet_number if unet_number is not None else self.num_unets
        self.validate_and_set_unet_being_trained(unet_number)
        self.set_accelerator_scaler(unet_number)

        index = unet_number - 1
        unet = self.unet_being_trained

        optimizer = getattr(self, f'optim{index}')
        scaler = getattr(self, f'scaler{index}')
        scheduler = getattr(self, f'scheduler{index}')
        warmup_scheduler = getattr(self, f'warmup{index}')

        # set the grad scaler on the accelerator, since we are managing one per u-net
        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(unet.parameters(), self.max_grad_norm)

        optimizer.step()
        optimizer.zero_grad()
        if self.use_ema:
            ema_unet = self.get_ema_unet(unet_number)
            ema_unet.update()

        # scheduler, if needed
        maybe_warmup_context = nullcontext() if not exists(warmup_scheduler) else warmup_scheduler.dampening()
        with maybe_warmup_context:
            # recommended in the docs
            if exists(scheduler) and not self.accelerator.optimizer_step_was_skipped:
                scheduler.step()

        self.steps += F.one_hot(torch.tensor(unet_number - 1, device=self.steps.device), num_classes=len(self.steps))

        if not exists(self.checkpoint_path):
            return

        total_steps = int(self.steps.sum().item())
        if total_steps % self.checkpoint_every:
            return
        self.save_to_checkpoint_folder()

    def print(self, msg):
        if not self.is_main:
            return
        if not self.verbose:
            return

        return self.accelerator.print(msg)

    def print_untrained_unets(self):
        print_final_error = False

        for ind, (steps, unet) in enumerate(zip(self.steps.tolist(), self.imagen.unets)):
            if steps > 0 or isinstance(unet, NullUnet):
                continue
            self.print(f'unet {ind + 1} has not been trained')
            print_final_error = True

        if print_final_error:
            self.print('when sampling, you can pass stop_at_unet_number to stop early in the cascade, '
                       'so it does not try to generate with untrained unets')

    def reset_ema_unets_all_one_device(self, device=None):
        if not self.use_ema:
            return

        device = default(device, self.device)
        self.ema_unets = nn.ModuleList([*self.ema_unets])
        self.ema_unets.to(device)
        self.ema_unet_being_trained_index = -1

    @torch.no_grad()
    @contextmanager
    def use_ema_unets(self):
        if not self.use_ema:
            output = yield
            return output

        self.reset_ema_unets_all_one_device()
        self.imagen.reset_unets_all_one_device()
        self.unets.eval()

        trainable_unets = self.imagen.unets
        # swap in exponential moving averaged unets for sampling
        self.imagen.unets = self.unets

        output = yield
        self.imagen.unets = trainable_unets  # restore original training unets
        # cast the ema_model unets back to original device
        for ema in self.ema_unets:
            ema.restore_ema_model_device()

        return output

    @torch.no_grad()
    @cast_torch_tensor
    @imagen_sample_in_chunks
    def sample(self, *args, **kwargs):
        context = nullcontext if kwargs.pop('use_non_ema', False) else self.use_ema_unets

        self.print_untrained_unets()

        if not self.is_main:
            kwargs['use_tqdm'] = False

        with context():
            output = self.imagen.sample(*args, device=self.device, **kwargs)

        return output
