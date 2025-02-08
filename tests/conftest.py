import logging
import os

import pytest

import pyroml as p
from pyroml.utils.env import PyroEnv, set_bool_env
from pyroml.utils.log import set_level_all_loggers

DEFAULT_SEED = 42


@pytest.fixture(scope="session", autouse=True)
def setup(log_level=logging.INFO):
    print("======== SETTING UP TEST ========")
    p.seed_everything(DEFAULT_SEED)

    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_PROJECT"] = "pyroml_test"

    set_level_all_loggers(log_level)
    set_bool_env(PyroEnv.VERBOSE, True)
