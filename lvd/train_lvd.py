import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from datasets.video_dataset import VideoData
from lvd.models.diffusion import LatentDiffusion

from config_utils import *


def main():
    pl.seed_everything(1234)
    all_configs = get_all_configs_by_parser()

    data_params = get_module_params("data", all_configs)
    data = VideoData(data_params)
    # pre-make relevant cached files if necessary
    data.train_dataloader()
    data.test_dataloader()

    model_config = all_configs.pop("model", OmegaConf.create())
    model = LatentDiffusion(**model_config.get("params", dict()))  # here I use default parameters to init the model

    call_backs = [ModelCheckpoint(monitor='val/loss', mode='min', save_top_k=-1)]

    trainer_params = get_module_params("trainer", all_configs)
    kwargs = dict()
    if trainer_params.gpus > 1:
        # find_unused_parameters = False to support gradient checkpointing
        kwargs = dict(gpus=trainer_params.gpus, plugins=[pl.plugins.DDPPlugin(find_unused_parameters=False)])
    trainer = pl.Trainer.from_argparse_args(callbacks=call_backs, max_steps=trainer_params.max_steps, **kwargs)

    trainer.fit(model, data)


if __name__ == '__main__':
    main()
