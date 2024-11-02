import sys

sys.path.append("..")

import os
import pyroml as p
import logging


DEFAULT_SEED = 42


def setup_test(log_level=logging.INFO):
    p.seed_everything(DEFAULT_SEED)

    # os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_PROJECT"] = "pyroml_test"

    p.set_level_all_loggers(log_level)
    p.set_bool_env(p.PyroEnv.VERBOSE, True)
