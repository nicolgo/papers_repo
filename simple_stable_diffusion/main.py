import os, sys, datetime, glob
import time

import ldm.modules.losses.vqperceptual
from config_utils import *

from omegaconf import OmegaConf

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer

from ldm.util import instantiate_from_config

if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError("-n/--name and -r/--resume cannot be specified both."
                         "If you want to resume training in a new log folder, "
                         "use -n/--name in combination with --resume_from_checkpoint")

    logdir, nowname = check_resume(opt, now)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)

        # lightning config
        lightning_config = config.pop("lightning", OmegaConf.create())
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        cpu = check_gpu_or_cpu(trainer_config)
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # model
        model = instantiate_from_config(config.model)

        # logger
        trainer_kwargs = configure_logger(logdir, model, lightning_config, ckptdir, nowname, trainer_opt, opt, config,
                                          cfgdir, now)

        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
        trainer.logdir = logdir

        # data
        data = instantiate_from_config(config.data)
        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
        data.prepare_data()
        data.setup()
        for k in data.datasets:
            print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

        configure_learning_rate(config, model, lightning_config, opt, cpu)

        # run
        if opt.train:
            try:
                trainer.fit(model, data)
            except Exception:
                raise
        if not opt.no_test and not trainer.interrupted:
            trainer.test(model, data)

    except:
        if opt.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
        if trainer.global_rank == 0:
            print(trainer.profiler.summary())
