import torch
import sys

print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("Device count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"Device {i} name:", torch.cuda.get_device_name(i))
        print(f"Device {i} capability:", torch.cuda.get_device_capability(i))
else:
    print("CUDA is not available in this PyTorch installation")

