### TODO:

- Checkpoints (+delete checkpoint folder if error and nothing stored in it)
- Finish loggers; generic + tensorboard + WandB
- Write proper tests
- add time per iteration, time per epoch, more?
- Add docs
- Add warnings properyl: https://github.com/Lightning-AI/utilities/blob/main/src/lightning_utilities/core/rank_zero.py
- use DistributeDataParallel:
  - ````py
    with ctx:
        with distributed_backend.get_context_for_microstep_forward(model=model, microstep_idx=microstep_idx, gradient_accumulation_steps=acc_steps):
            outputs = model(x, targets=y)```
    ````
- fused adam ?
```py
fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
use_fused = fused_available and device_type == 'cuda'
extra_args = dict(fused=True) if use_fused else dict()
optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
```
- amp GradScaler for float16
- Add benchmarks callbacks
  - Memory 
```python
torch.cuda.memory._record_memory_history(enabled='all')
s = torch.cuda.memory._snapshot()
with open(f"snapshot.pickle", "wb") as f:
    dump(s, f)
torch.cuda.memory._record_memory_history(enabled=None)
```
  