import torch
from setup import setup_test

import pyroml as p


class Model(p.PyroModel):
    def __init__(self, a=12):
        super().__init__()
        self.a = a
        self.b = torch.rand(a)

        self.register_hparams("a", "b")

        self.register_buffer("c", torch.arange(a))  # Saved with model weights
        self.d = 42  # not hparams


if __name__ == "__main__":
    setup_test()

    m = Model()
    a, b, c, d = m.a, m.b, m.c, m.d

    checkpoint_folder = "./checkpoints/save-load-test"
    m.save(checkpoint_folder=checkpoint_folder)

    m.a = -1
    m.b = torch.rand(a)
    m.c = torch.arange(a) * -1
    m.d = -1

    m.load(checkpoint_folder=checkpoint_folder)

    assert m.a == a
    assert torch.eq(m.b, b).all()
    assert torch.eq(m.c, c).all()
    assert m.d != d
