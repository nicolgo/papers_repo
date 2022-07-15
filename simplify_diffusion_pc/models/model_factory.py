from .vae_diffusion import *
from .vae_flow import *
from .vae_gaussian import *


def get_model_by_type(model_type, arguments):
    gen_model = None
    if model_type == 'gaussian':
        gen_model = GaussianVAE(arguments)
    elif model_type == 'flow':
        gen_model = FlowVAE(arguments)
    elif model_type == 'pure':
        gen_model = PureDiffusion(arguments)
    else:
        pass
    return gen_model
