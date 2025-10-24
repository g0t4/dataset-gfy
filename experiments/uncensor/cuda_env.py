import os

def use_6000() -> None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def use_5090() -> None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def list_gpus() -> None:
    """ FYI this is cached, so run this after setting use_ above:"""
    import torch

    print("GPU list:")
    for i in range(torch.cuda.device_count()):
        print(f"    {i}: {torch.cuda.get_device_name(i)}")
