#!/usr/bin/env python3

# Copyright 2022 Brno University of Technology (author: Federico Landini)
# Licensed under the MIT license.

from safe_gpu import safe_gpu


def use_single_gpu(gpus_qty: int) -> safe_gpu.GPUOwner: #def dummy():
    assert gpus_qty < 2, "Multi-GPU still not available."
    gpu_owner = safe_gpu.GPUOwner(nb_gpus=gpus_qty)
    return gpu_owner

def use_multi_gpu():
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU instead.")
        return torch.device("cpu"), None  # Use CPU

    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs!")

    device = torch.device("cuda")
    return device, num_gpus

def setup_model_for_gpus(model):
    device, num_gpus = use_multi_gpu()
    
    model = model.to(device)  # Move model to GPU
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)  # Provide multi-GPU support

    return model, device