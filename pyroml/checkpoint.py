import os
import torch

from enum import Enum
from json import JSONEncoder


class EncodeTensor(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return super(EncodeTensor, self).default(obj)


# TODO: move this to status  class
"""
self.date = get_date()  

def get_folder(self):
    # TODO: folder Based on version number as in lightning
    folder = os.path.join(
        self.config.checkpoint_folder,
        f"{self.date}_epoch={self.epoch:03d}_iter={self.iteration:06d}",
    )
    os.makedirs(folder, exist_ok=True)
    return folder"""


class Checkpoint(Enum):
    MODEL_WEIGHTS = "model_weights.safetensors"
    # FIXME: should model_state even exist? Store optimizer / scheduler to the trainer state
    MODEL_STATE = "model_state.pt"
    MODEL_HPARAM = "model_hparams.json"

    # TODO: separate config and training file
    # NOTE: Probably the status class should handle saving the trainer state
    TRAINER_STATE = "training_state.pt"
