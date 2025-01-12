### TODO:

- add docs
- use DistributeDataParallel:
  - ````py
    with ctx:
        with distributed_backend.get_context_for_microstep_forward(model=model, microstep_idx=microstep_idx, gradient_accumulation_steps=acc_steps):
            outputs = model(x, targets=y)```
    ````
- add time per iteration, time per epoch, more?
- fused adam ?
```py
fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
use_fused = fused_available and device_type == 'cuda'
extra_args = dict(fused=True) if use_fused else dict()
optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
```
- amp GradScaler for float16
