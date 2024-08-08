import sys

sys.path.append("..")

import os
import pyroml as p
import logging


DEFAULT_SEED = 42


def setup_test():
    p.seed_everything(DEFAULT_SEED)

    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_PROJECT"] = "pyroml_test"

    p.set_level_all_loggers(logging.DEBUG)
    p.set_bool_env(p.PyroEnv.VERBOSE, True)
