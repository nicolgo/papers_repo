import argparse
import os
import datetime


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
    parser.add_argument("-s", "--seed", type=int, default=23, help="seed for seed_everything", )
    parser.add_argument("-l", "--logdir", type=str, default="logs", help="directory for logging dat shit", )
    parser.add_argument("-t", "--train", type=str2bool, const=True, default=False, nargs="?", help="train", )
    return parser


def config_resume(opt):
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
    else:
        if opt.name:
            name = "_" + opt.name
        else:
            name = ""
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        postfix_name = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, postfix_name)

    return logdir
