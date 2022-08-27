import numpy as np
import torch as th


def normal_kl(mean1, logvar1, mean2, logvar2):
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None

    logvar1, logvar2 = [x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor) for x in (logvar1, logvar2)]
    # see https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    return 0.5 * (-1 + logvar2 - logvar1 + th.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * th.exp(-logvar2))
