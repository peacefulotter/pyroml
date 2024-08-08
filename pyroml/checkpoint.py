from enum import Enum

# import torch
# from json import JSONEncoder
# class EncodeTensor(JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, torch.Tensor):
#             return obj.cpu().detach().numpy().tolist()
#         return super(EncodeTensor, self).default(obj)


class Checkpoint(Enum):
    MODEL_WEIGHTS = "weights.safetensors"
    MODEL_HPARAMS = "model_hparams.json"
    TRAINER_STATE = "training_state.pt"
    TRAINER_HPARAMS = "trainer_hparams.json"
