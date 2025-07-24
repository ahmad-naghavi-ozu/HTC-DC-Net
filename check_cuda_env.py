import os
import sys
import torch

print("=== Python and PyTorch Information ===")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch CUDA compiled version: {torch.version.cuda}")
print(f"PyTorch CUDA is available: {torch.cuda.is_available()}")

print("\n=== CUDA Environment Variables ===")
cuda_vars = [
    "CUDA_HOME", 
    "CUDA_PATH", 
    "CUDA_VISIBLE_DEVICES", 
    "CUDA_DEVICE_ORDER",
    "LD_LIBRARY_PATH"
]
for var in cuda_vars:
    print(f"{var}: {os.environ.get(var, 'Not set')}")

print("\n=== PyTorch CUDA Configuration ===")
if torch.cuda.is_available():
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    print(f"Current CUDA Device: {torch.cuda.current_device()}")
    print(f"Current CUDA Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("PyTorch CUDA is not available")
    
    # Try to find CUDA libraries
    print("\n=== Trying to locate CUDA libraries ===")
    possible_cuda_paths = [
        "/usr/local/cuda",
        "/usr/local/cuda-12.5",
        "/usr/local/cuda-12.1",
        "/usr/local/cuda-11.8",
    ]
    
    for path in possible_cuda_paths:
        if os.path.exists(path):
            print(f"Found CUDA installation at: {path}")
            # Check if libcudart.so exists
            lib_path = os.path.join(path, "lib64")
            if os.path.exists(lib_path):
                libs = [f for f in os.listdir(lib_path) if "libcudart.so" in f]
                if libs:
                    print(f"CUDA Runtime libraries found at {lib_path}: {libs}")
            # Check NVCC version
            nvcc_path = os.path.join(path, "bin/nvcc")
            if os.path.exists(nvcc_path):
                print("NVCC found, checking version...")
                try:
                    import subprocess
                    result = subprocess.run([nvcc_path, "--version"], capture_output=True, text=True)
                    print(result.stdout)
                except Exception as e:
                    print(f"Error checking NVCC version: {e}")

print("\n=== PyTorch Library Paths ===")
print(f"PyTorch Library Path: {torch.__file__}")
print(f"Is PyTorch built with CUDA: {'not torch._C._GLIBCXX_USE_CXX11_ABI' not in dir(torch._C)}")

# Check if PyTorch can create a CUDA tensor
print("\n=== Testing CUDA Tensor Creation ===")
try:
    x = torch.tensor([1.0, 2.0, 3.0]).cuda()
    print("✓ Successfully created a CUDA tensor")
    print(f"Tensor device: {x.device}")
except Exception as e:
    print(f"✗ Failed to create CUDA tensor: {e}")

# Print CUDA library load paths
print("\n=== CUDA Library Search Paths ===")
try:
    import ctypes
    cuda = ctypes.CDLL("libcuda.so", mode=ctypes.RTLD_GLOBAL)
    print("✓ Successfully loaded libcuda.so")
except Exception as e:
    print(f"✗ Failed to load libcuda.so: {e}")

# Print conda environment info
print("\n=== Conda Environment Info ===")
conda_prefix = os.environ.get("CONDA_PREFIX", "Not in conda environment")
print(f"Conda prefix: {conda_prefix}")
if conda_prefix != "Not in conda environment":
    conda_cuda_path = os.path.join(conda_prefix, "lib")
    if os.path.exists(conda_cuda_path):
        cuda_libs = [f for f in os.listdir(conda_cuda_path) if "cuda" in f.lower() or "nvidia" in f.lower()]
        if cuda_libs:
            print(f"Found CUDA-related libraries in conda env: {cuda_libs[:10]}")
            if len(cuda_libs) > 10:
                print(f"... and {len(cuda_libs) - 10} more")

# Print information about NVIDIA NVML
print("\n=== NVIDIA Management Library (NVML) ===")
try:
    import pynvml
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    print(f"NVML Device Count: {device_count}")
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle)
        print(f"GPU {i}: {name}")
    pynvml.nvmlShutdown()
except ImportError:
    print("pynvml not installed")
except Exception as e:
    print(f"Error using NVML: {e}")