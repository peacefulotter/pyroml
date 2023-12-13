### TODO:

- seed
- add docs
- Move most of the Trainer **init** code to the run method, to allow modifying ALL configs between runs
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
- add time per iteration, time per epoch, more?
