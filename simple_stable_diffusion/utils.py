import argparse, os, glob
from pytorch_lightning.trainer import Trainer

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-n", "--name", type=str, const=True, default="", nargs="?", help="postfix for logdir", )
    parser.add_argument("-r", "--resume", type=str, const=True, default="", nargs="?",
                        help="resume from logdir or checkpoint in logdir", )
    parser.add_argument("-b", "--base", nargs="*", metavar="base_config.yaml",
                        help="""paths to base configs. Loaded from left-to-right. Parameters can be 
                        overwritten or added with command-line options of the form `--key value`.""",
                        default=list(), )
    parser.add_argument("-t", "--train", type=str2bool, const=True, default=False, nargs="?", help="train", )
    parser.add_argument("--no-test", type=str2bool, const=True, default=False, nargs="?", help="disable test", )
    parser.add_argument("-p", "--project", help="name of new or path to existing project")
    parser.add_argument("-d", "--debug", type=str2bool, nargs="?", const=True, default=False,
                        help="enable post-mortem debugging", )
    parser.add_argument("-s", "--seed", type=int, default=23, help="seed for seed_everything", )
    parser.add_argument("-f", "--postfix", type=str, default="", help="post-postfix for default name", )
    parser.add_argument("-l", "--logdir", type=str, default="logs", help="directory for logging dat shit", )
    parser.add_argument("--scale_lr", type=str2bool, nargs="?", const=True, default=True,
                        help="scale base-lr by ngpu * batch_size * n_accumulate", )
    return parser


def check_resume(opt, now):
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)
    return logdir, nowname


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))

def check_gpu_or_cpu(trainer_config):
    if not "gpus" in trainer_config:
        del trainer_config["accelerator"]
        cpu = True
    else:
        print(f"Running on GPUs {trainer_config['gpus']}")
        cpu = False
    return cpu
