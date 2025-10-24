import einops
from einops import rearrange
import torch
from torch import tensor

import kata.helpers
from kata.helpers import assert_close

order_quantities = tensor([1.0, 4.0, 2.0])

# add innermost dimension!
kata.helpers.DEBUG = True
assert_close( \
    rearrange(order_quantities, "a -> a 1"), \
    order_quantities.unsqueeze(1), \
)

# %% add outermost dimension

assert_close( \
    rearrange(order_quantities, "a -> 1 a"), \
    order_quantities.unsqueeze(0), \
)

# %%

