"""S0 GPU trainer: PyTorch mirror of the reference transformer; correctness
defined by parity with the NumPy oracle. S1 next: reference dataset wiring, DDP."""
from tinymind_gpu.model import TorchTinyMind
from tinymind_gpu.parity import copy_numpy_to_torch, copy_torch_to_numpy, forward_parity, grad_parity
__all__ = ["TorchTinyMind", "copy_numpy_to_torch", "copy_torch_to_numpy", "forward_parity", "grad_parity"]
