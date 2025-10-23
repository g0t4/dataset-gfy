import torch
from rich import print

global DEBUG
DEBUG = False

def assert_close(
    left: torch.Tensor | object,
    right: torch.Tensor | object,
    *,
    rtol: float = 1e-05,
    atol: float = 1e-08,
) -> None:

    if not torch.is_tensor(left):
        if isinstance(left, (float, int, str)):
            left = torch.Tensor([left])
        elif isinstance(left, list):
            left = torch.Tensor(left)

    if not torch.is_tensor(right):
        if isinstance(right, (float, int, str)):
            right = torch.Tensor([right])
        elif isinstance(right, list):
            right = torch.Tensor(right)

    if not left.dtype == right.dtype:
        if DEBUG:
            print(f"[WARN] dtype mismatch, will attempt conversion: {left.dtype=} vs {right.dtype=}")
        right = right.to(left.dtype)

    if not torch.allclose(left, right, rtol=rtol, atol=atol):
        if DEBUG:
            print(f"mismatch: {left=} | {right=}")
        raise AssertionError(f" tensors aren't close, unlike your mom! {left} vs {right}")

    if DEBUG:
        print(f"[DEBUG]: {left=} | {right=}")
