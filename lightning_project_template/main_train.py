import os
from root_dir import PROJECT_DIR
import pytorch_lightning as pl
from config_utils import *
from omegaconf import OmegaConf
from utils.util import instantiate_from_config

if __name__ == "__main__":
    out_dir = PROJECT_DIR + os.sep + "outputs"
    log_dir = PROJECT_DIR + os.sep + "logs"
    print(out_dir)
    print(log_dir)
    parser = get_parser()
    parser = pl.Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    pl.seed_everything(opt.seed)

    # init and save configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    try:
        # logger
        trainer = pl.Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
        trainer.logdir = logdir
        # data
        data = instantiate_from_config(config.data)
        data.prepare_data()
        data.setup()
        # model
        model = instantiate_from_config(config.model)
        # run
        if opt.train:
            try:
                trainer.fit(model, data)
            except Exception:
                raise
        if not opt.no_test and not trainer.interrupted:
            trainer.test(model, data)
    except:
        if opt.debug:
            pass
        raise
    finally:
        pass
