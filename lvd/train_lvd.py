import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from datasets.video_dataset import VideoData
from lvd.models.diffusion import LatentDiffusion

from config_utils import *


def main():
    pl.seed_everything(1234)
    all_configs, trainer_opt = get_all_configs_by_parser()

    data_params = get_module_params("data", all_configs)
    data = VideoData(data_params)
    # pre-make relevant cached files if necessary
    data.train_dataloader()
    data.test_dataloader()

    model_config = all_configs.pop("model", OmegaConf.create())
    model = LatentDiffusion(**model_config.get("params", dict()))  # here I use default parameters to init the model
    model.learning_rate = model_config.base_learning_rate

    trainer_kwargs = dict()
    call_backs = [ModelCheckpoint(monitor='val/loss', mode='min')]
    trainer_kwargs["callbacks"] = call_backs
    trainer = pl.Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)

    trainer.fit(model, data)


if __name__ == '__main__':
    main()
