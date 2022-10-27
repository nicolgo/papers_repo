import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from datasets.video_dataset import VideoData
from lvd.models.diffusion import LatentDiffusion


def main():
    pl.seed_everything(1234)

    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--data_path', type=str, default="data/ucf101")
    parser.add_argument('--resolution', type=int, default=128)
    parser.add_argument('--sequence_length', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    data = VideoData(args)
    # pre-make relevant cached files if necessary
    data.train_dataloader()
    data.test_dataloader()

    model = LatentDiffusion()  # here I use default parameters to init the model

    call_backs = [ModelCheckpoint(monitor='val/loss', mode='min', save_top_k=-1)]

    kwargs = dict()
    if args.gpus > 1:
        # find_unused_parameters = False to support gradient checkpointing
        kwargs = dict(gpus=args.gpus, plugins=[pl.plugins.DDPPlugin(find_unused_parameters=False)])
    trainer = pl.Trainer.from_argparse_args(args, callbacks=call_backs,
                                            max_steps=args.max_steps, **kwargs)

    trainer.fit(model, data)


if __name__ == '__main__':
    main()
