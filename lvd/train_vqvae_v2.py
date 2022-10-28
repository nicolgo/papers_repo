import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from datasets.video_dataset import VideoData
from lvd.models.vqvae import VQVAE
from omegaconf import OmegaConf


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--base", nargs="*", metavar="base_config.yaml",
                        help="paths to base configs. Loaded from left-to-right. ", default=list(), )
    parser = pl.Trainer.add_argparse_args(parser)
    opt, unknown = parser.parse_known_args()

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    all_configs = OmegaConf.merge(*configs, cli)

    return all_configs


def get_trainer_configs(all_configs):
    lightning_config = all_configs.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    return trainer_config


def main():
    pl.seed_everything(1234)
    all_configs = get_parser()
    data_config = all_configs.pop("data", OmegaConf.create())
    data_params = data_config["params"]

    # parser = argparse.ArgumentParser()
    # parser = pl.Trainer.add_argparse_args(parser)
    # parser = VQVAE.add_model_specific_args(parser)
    # parser.add_argument('--data_path', type=str, default='/home/wilson/data/datasets/bair.hdf5')
    # parser.add_argument('--sequence_length', type=int, default=16)
    # parser.add_argument('--resolution', type=int, default=64)
    # parser.add_argument('--batch_size', type=int, default=32)
    # parser.add_argument('--num_workers', type=int, default=8)
    # args = parser.parse_args()

    data = VideoData(data_params)
    data.train_dataloader()
    data.test_dataloader()

    model_config = all_configs.pop("model", OmegaConf.create())
    model_params = model_config["params"]
    model = VQVAE(model_params)

    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='val/recon_loss', mode='min'))

    trainer_config = get_trainer_configs(all_configs)
    kwargs = dict()
    if trainer_config.gpus > 1:
        kwargs = dict(distributed_backend='ddp', gpus=trainer_config.gpus)
    trainer = pl.Trainer.from_argparse_args(callbacks=callbacks,
                                            max_steps=200000, **kwargs)

    trainer.fit(model, data)


if __name__ == '__main__':
    main()
