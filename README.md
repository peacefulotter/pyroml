# pyroml

🔥 Machine Learning tool allowing plug-and-play training for pytorch models

### Installation

```shell
$ git clone https://github.com/peacefulotter/pyroml.git
$ cd pyroml
$ sudo apt install python3.10-venv # check you python version and change it here if !=
$ sudo apt install python3-virtualenv
$ python3 -m venv venv
$ source ./venv/bin/activate
$ pip install -r requirements.txt
```

#### Running tests

```shell
$ cd tests
$ python main.py # this will launch the training, goto https://wandb.ai/otters-gang/pyro_main_test/workspace  to see the training occuring (should be really fast)
$ python pretrain.py # will load the last checkpoint and compute mse on a small part of the dataset, outputs True if model predicts correctly!
```

### Done

- Metrics, with support for custom metrics
- WandB
- Checkpoints
- Load pretrained models from checkpoints

### TODO:

- seed
- use ctx:
  - type_ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
    for auto compatibility between model output and dataset target
    https://github.com/epfml/llm-baselines/blob/main/src/optim/base.py#L14C5-L15C56
