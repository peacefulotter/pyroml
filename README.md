# 🔥 pyro

<b style='font-size:16px'>Lightweight Machine Learning framework allowing plug-and-play training for Pytorch models</b>

-   ⚡ <b>Lightning</b> inspired
-   💾 Support for <b>wandb</b> and <b>checkpoints</b> out of the box
-   📊 Pretty <b>logs</b> and out-of-the-box <b>metrics</b> support
-   ✨ Fully <b>type-safe</b>
-   🪶 Lightweight and <b>easy to use</b>

## Requirements

-   Python 3.10 : 3.12

## Installation

```shell
pip install pyroml
```

### Locally

```shell
git clone https://github.com/peacefulotter/pyroml.git
cd pyroml

poetry config virtualenvs.in-project true
```

### Running tests

```shell
$ cd tests
$ python main.py # this will launch the training, follow the wandb link to access the plots
$ python pretrain.py # will load the last checkpoint and compute mse on a small part of the dataset, outputs True if model predicts correctly!
```
