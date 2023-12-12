### TODO:

- rename Trainer.run to Trainer.fit
- seed
- add docs
- use ctx:
  - ````py
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
    ```og
    ````
- use DistributeDataParallel:
  - ````py
    with ctx:
        with distributed_backend.get_context_for_microstep_forward(model=model, microstep_idx=microstep_idx, gradient_accumulation_steps=acc_steps):
            outputs = model(x, targets=y)```
    ````
- add logs at the beginning of training
- add time per iteration, time per epoch, more?
