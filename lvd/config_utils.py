import argparse
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf


def get_all_configs_by_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--base", nargs="*", metavar="base_config.yaml",
                        help="paths to base configs. Loaded from left-to-right. ", default=list(), )
    parser = pl.Trainer.add_argparse_args(parser)
    opt, unknown = parser.parse_known_args()

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    all_configs = OmegaConf.merge(*configs, cli)

    trainer_params = get_module_params("trainer", all_configs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "devices" in trainer_params and device.type == "cuda":
        trainer_params["accelerator"] = 'gpu'
        gpuinfo = trainer_params["devices"]
        print(f"Running on GPUs {gpuinfo}")
    else:
        trainer_params["accelerator"] = 'cpu'
    trainer_opt = argparse.Namespace(**trainer_params)
    return all_configs, trainer_opt


def get_module_params(module_name, configs):
    config = configs.pop(module_name, OmegaConf.create())
    params = config["params"]
    return params
