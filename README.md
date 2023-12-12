# pyroml

🔥 Machine Learning framework allowing plug-and-play training for pytorch models

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
$ python main.py # this will launch the training, follow the wandb link to access the plots
$ python pretrain.py # will load the last checkpoint and compute mse on a small part of the dataset, outputs True if model predicts correctly!
```

### Done

- Metrics, with support for custom metrics
- WandB
- Checkpoints
- Load pretrained models from checkpoints
