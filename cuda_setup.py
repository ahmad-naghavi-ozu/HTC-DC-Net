import os
import torch
import sys

def setup_cuda(gpu_ids="0"):
    """
    Configure CUDA environment variables and return device setup
    
    Args:
        gpu_ids (str): Comma-separated GPU indices to use (e.g., "0,1,2")
    
    Returns:
        tuple: (device, gpu_list)
            - device: torch.device to use
            - gpu_list: list of GPU indices to use, empty if using CPU
    """
    # Set up GPU environment variables for CUDA device management
    # Ensure CUDA devices are ordered by PCI bus ID for consistent behavior across sessions
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    # Specify which GPU(s) to use
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    
    # Set TensorFlow log level to a specific level (if TensorFlow is used elsewhere)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all messages, 1 = filter out INFO, 2 = filter out INFO & WARNINGS, 3 = only ERROR messages
    
    # Parse GPU IDs into a list
    gpu_list = []
    if gpu_ids and gpu_ids.strip():
        gpu_list = [int(gpu_id.strip()) for gpu_id in gpu_ids.split(",") if gpu_id.strip()]
    
    # Check CUDA availability for PyTorch
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"CUDA is available. Using GPU(s): {gpu_ids}")
        
        is_multi_gpu = len(gpu_list) > 1
        if is_multi_gpu:
            print(f"Multi-GPU setup detected. Using {len(gpu_list)} GPUs: {gpu_list}")
            print(f"GPU devices available: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print(f"Using single GPU: {gpu_ids}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("CUDA is not available. Using CPU instead.")
    
    return device, gpu_list