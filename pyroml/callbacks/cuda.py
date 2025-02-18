from pyroml.callbacks.callback import Callback


class CudaStats(Callback):
    pass


# https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/lightning/pytorch/accelerators/cuda.py#L34

# def _get_gpu_id(device_id: int) -> str:
#     """Get the unmasked real GPU IDs."""
#     # All devices if `CUDA_VISIBLE_DEVICES` unset
#     default = ",".join(str(i) for i in range(num_cuda_devices()))
#     cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", default=default).split(",")
#     return cuda_visible_devices[device_id].strip()

#     # nvidia_smi_path = shutil.which("nvidia-smi")
# if nvidia_smi_path is None:
#     raise FileNotFoundError("nvidia-smi: command not found")

# gpu_stat_metrics = [
#     ("utilization.gpu", "%"),
#     ("memory.used", "MB"),
#     ("memory.free", "MB"),
#     ("utilization.memory", "%"),
#     ("fan.speed", "%"),
#     ("temperature.gpu", "°C"),
#     ("temperature.memory", "°C"),
# ]
# gpu_stat_keys = [k for k, _ in gpu_stat_metrics]
# gpu_query = ",".join(gpu_stat_keys)

# index = torch._utils._get_device_index(device)
# gpu_id = _get_gpu_id(index)
# result = subprocess.run(
#     [nvidia_smi_path, f"--query-gpu={gpu_query}", "--format=csv,nounits,noheader", f"--id={gpu_id}"],
#     encoding="utf-8",
#     capture_output=True,
#     check=True,
# )
