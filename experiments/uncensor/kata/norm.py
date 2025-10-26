import math
from torch import tensor
import torch
from rich import print
from kata.helpers import *
import kata.helpers

kata.helpers.DEBUG = True

simple = tensor([1, 3, 6], dtype=torch.float16)
# simple.sum() == 10
assert_close(simple.sum(), 10)
assert_close(simple.norm(), math.sqrt(46))  # magnitude => sqrt(1^2+3^2+6^2) == sqrt(46) ~= 6.78232998

# %% note error from lower precision float16 vs float32

simple_float16 = tensor([1, 3, 6], dtype=torch.float16)
simple_float32 = tensor([1, 3, 6], dtype=torch.float32)
assert_close(simple_float16.norm(), math.sqrt(46.0))  # 6.7812 # note rounding error is material here in 3rd decimal and after
assert_close(simple_float32.norm(), math.sqrt(46.0))  # 6.7823 (matches hand calculation of sqrt(46))

# %%

third = tensor(
    [
        [
            [2, 1, 3],
            [0, 1, -1],
        ],  # sum==6 count=6 => mean=1
        [
            [3, 6, 4],
            [0, 1, -2],
        ],  #sum==12 count=6 => mean=2
    ],
    dtype=torch.float32)

assert_close(third.norm(), 9.05538514)

# %%
norm_dim2 = tensor([[
    math.sqrt(4 + 1 + 9),
    math.sqrt(0 + 1 + 1),
], [
    math.sqrt(3 * 3 + 6 * 6 + 4 * 4),
    math.sqrt(0 + 1 + 4),
]])

assert_close(third.norm(dim=2), norm_dim2)
assert_close(third.norm(dim=-1), norm_dim2)  # same

# %%

# [
#     [
#         [2, 1, 3],
#         [0, 1, -1],
#     ].norm(dim=0),
#     [
#         [3, 6, 4],
#         [0, 1, -2],
#     ].norm(dim=0),
# ],

norm_dim1 = tensor([
    [math.sqrt(2 * 2 + 0 * 0), math.sqrt(1 * 1 + 1 * 1), math.sqrt(3 * 3 + -1 * -1)],
    [math.sqrt(3 * 3 + 0 * 0), math.sqrt(6 * 6 + 1 * 1), math.sqrt(4 * 4 + -2 * -2)],
])

assert_close(third.norm(dim=1), norm_dim1)

# %%
# [
#     [
#         [2, 1, 3],
#         [0, 1, -1],
#     ],
#     [
#         [3, 6, 4],
#         [0, 1, -2],
#     ],
# ].norm(dim=0),

norm_dim0 = tensor([
    [math.sqrt(2 * 2 + 3 * 3), math.sqrt(1 * 1 + 6 * 6), math.sqrt(3 * 3 + 4 * 4)],
    [math.sqrt(0 * 0 + 0 * 0), math.sqrt(1 * 1 + 1 * 1), math.sqrt(-1 * -1 + -2 * -2)],
])
assert_close(third.norm(dim=0), norm_dim0)
