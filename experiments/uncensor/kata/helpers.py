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
            left = torch.tensor([left])
        elif isinstance(left, list):
            left = torch.tensor(left)
        elif isinstance(left, torch.Size):
            print(f'{left=}')
            print("size")
            left = torch.tensor(left)
            print(f'{left=}')
        elif isinstance(left, tuple):
            left = torch.tensor(left)
        else:
            raise TypeError(f"left type {type(left)} not implemented yet.")

    if not torch.is_tensor(right):
        if isinstance(right, (float, int, str)):
            right = torch.tensor([right])
        elif isinstance(right, list):
            right = torch.tensor(right)
        elif isinstance(right, torch.Size):
            right = torch.tensor(right)
        elif isinstance(right, tuple):
            right = torch.tensor(right)
        else:
            raise TypeError(f"right type {type(right)} not implemented yet.")

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
