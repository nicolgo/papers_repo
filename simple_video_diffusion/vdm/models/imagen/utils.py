import torch
from functools import wraps
import torch.nn.functional as F
import torch.nn as nn


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cast_uint8_images_to_float(images):
    if not images.dtype == torch.uint8:
        return images
    return images / 255


def maybe(fn):
    @wraps(fn)
    def inner(x):
        if not exists(x):
            return x
        return fn(x)

    return inner


def cast_tuple(val, length=None):
    if isinstance(val, list):
        val = tuple(val)

    output = val if isinstance(val, tuple) else ((val,) * default(length, 1))

    if exists(length):
        assert len(output) == length

    return output


def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


def resize_image_to(image, target_image_size, clamp_range=None):
    orig_image_size = image.shape[-1]

    if orig_image_size == target_image_size:
        return image

    out = F.interpolate(image, target_image_size, mode='nearest')

    if exists(clamp_range):
        out = out.clamp(*clamp_range)

    return out


def normalize_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_zero_to_one(normed_img):
    return (normed_img + 1) * 0.5


def identity(t, *args, **kwargs):
    return t


