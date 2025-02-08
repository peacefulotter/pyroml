import os
import random

import numpy as np
import torch

from pyroml.utils.log import get_logger


def seed_everything(seed):
    log = get_logger(__name__)
    log.info(f"Global seed set to {seed}")

    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
