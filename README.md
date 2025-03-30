# 🔥 pyro

<b style='font-size:16px'>Lightweight Machine Learning framework allowing plug-and-play training for Pytorch models</b>

- ⚡ <b>Lightning</b> inspired
- 💾 Support for <b>wandb</b> and <b>checkpoints</b> out-of-the-box
- 📊 Pretty <b>logs</b>, <b>plots</b> and support for <b>metrics</b>
- ✨ Fully <b>type-safe</b>
- 🪶 Lightweight and <b>easy to use</b>

## Examples

See [📓 notebooks](https://github.com/peacefulotter/pyroml/blob/main/notebooks/) for examples using pyro. In particular, you can find:

- **[Iris](https://github.com/peacefulotter/pyroml/blob/main/notebooks/iris.ipynb)** : Simplest example training a small MLP on the Iris dataset.
- **[SmolVLM on Flowers102](https://github.com/peacefulotter/pyroml/blob/main/notebooks/smolvlm-flowers102.ipynb)** : Features from [SmolVLM](https://github.com/huggingface/smollm) vision model are extracted and used to train a linear classifier on the Flowers102 dataset, reaching a test accuracy of 98.6%.

## Usage

You can use 🔥 _pyro_ with minimal code changes and forever forget about writing training loops. Here is an example of a pyro model and training script to get you started.

### 1. Define your **Model**

```py
import torch
import pyroml as p

class MySOTAModel(p.PyroModule):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Linear(16, 1)
        self.loss_fn = torch.nn.MyLossFunction()

    # As you would do in your nn.Module class
    def forward(self, x):
        return self.model(x)

    # Optionally, configure your own optimizer and scheduler, see more in the docs
    def configure_optimizers(self, args):
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=args.trainer.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=0.99
        )

    def step(self, batch, stage: p.Stage):
        # Extract data from your dataset batch
        # Batches and model are moved to the appropriate device automatically
        x, y = batch
        # Forward the model
        preds = self(x)
        # Compute the loss
        loss = self.loss_fn(preds, y)
        # Optionally, register some metrics
        self.log(loss=loss.item(), accuracy=compute_accuracy(preds, y))
        # Return loss when training, otherwise return predictions
        if stage == p.Stage.TRAIN:
            return loss
        return preds

```

### 2. Instantiate a **Trainer**

```py
trainer = p.Trainer(
    lr=0.01,
    max_epochs=32,
    batch_size=16,
    # And many other options such as device, precision, callbacks, ...
)
```

### 3. Run **training**, **validation** and **testing**

```py
# Fit the model on given training set and evaluate the model during training
train_tracker = trainer.fit(model, training_dataset, validation_dataset)
print(train_tracker.records)

# Plot metric curves registered during training
train_tracker.plot(epoch=True)

# Evaluate your model after training
validation_tracker = trainer.evaluate(model, validation_dataset)
print(validation_tracker.records)

# Test your model on some testing set
_, test_preds = trainer.predict(model, test_dataset)
print("Test Predictions", test_preds)
```

## 🐍 Requirements

- **Python** ^3.10 | ^3.11 | ^3.12
- Recommended, either:
  - **uv** ^0.6 ([docs](https://docs.astral.sh/uv/))
  - **poetry** ^2 ([docs](https://python-poetry.org/docs/))

## 📦 Installation

Before proceeding, we suggest taking a quick look at the [`pyproject.toml`](pyproject.toml) for extra groups that you might want. Below are package manager specific commands to install pyro along with its dependencies.

### uv

```sh
# For default installation
uv add pyroml
# Add an extra group to target specific pytorch installs, use either cpu or cu124
uv add pyroml --extra (cpu / cu124)
# To also install additional dependencies that you might require
uv add pyroml --extra extra
```

### poetry

```sh
# CPU only version
poetry add pyroml
# OR with CUDA-enabled PyTorch and torchvision
poetry add pyroml[cuda] --source pytorch-cu124
# To also install additional dependencies that you might require
poetry add [...] --extras extra
```

### pip

```sh
# CPU only version
pip install pyroml
# OR with CUDA-enabled PyTorch and torchvision
pip install pyroml[cuda]
# To also install additional dependencies that you might require
pip install pyroml[extra]
```

### Locally

```sh
# Clone the repo
git clone https://github.com/peacefulotter/pyroml.git
cd pyroml
```

#### uv

```sh
# For basic installation, add an extra group to target specific pytorch installs
uv sync (--extra cpu / cu124)
# To also install additional packages
uv sync --extra extra
# To install all possibly required packages
uv sync --extra full
```

#### Poetry

```sh
# Install dependencies
poetry config virtualenvs.in-project true
poetry install --with cpu,dev # ,cuda
```

## Tests

Running tests has been made easy using pytest. First install the package and run the script:

```sh
poetry install --with test
./run_tests.sh
```
