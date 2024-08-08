import torch
import torch.nn as nn
import safetensors.torch as st

from .setup import setup_test

import pyroml as p


class Model(p.PyroModel):
    def __init__(self):
        super().__init__()
        self.register_buffer("test", torch.arange(12))


if __name__ == "__main__":
    setup_test()

    m = Model()
    print(m.test)
    st.save_model(m, "temp.st")
    m.test = torch.rand(12)
    print(m.test)
    st.load_model(m, "temp.st")
    print(m.test)
