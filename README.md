# pyro

🔥 Machine Learning tool allowing plug-and-play training for pytorch models

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
