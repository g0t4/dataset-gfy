import pytest
import torch
from torch import tensor

from kata.helpers import assert_close
import kata.helpers

# kata.helpers.DEBUG = True

a = tensor([1, 2, 4])
b = tensor([0, 1, 0])
c = tensor([5, 0, 0])

a * b  # tensor([0, 2, 0])
a * c  # tensor([5, 0, 0])
c * b  # tensor([0, 0, 0])

# FYI the ipython REPL shows each line before any failures so that's PERFECT traceability
#  AND assert_close literally explains what is different! so you don't even have to look at the tensors and compare
assert_same = torch.testing.assert_close
assert_same(a * b, torch.tensor([0, 2, 0]))
assert_same(a * c, torch.tensor([5, 0, 0]))
assert_same(c * b, torch.tensor([0, 0, 0]))

# %%

a.dot(b)  # 2
a.dot(c)  # 5
c.dot(b)  # 0

# %%

a
b
c

# manual dot prod
def manual_dot_product(left, right):
    assert len(left) == len(right)
    sum = 0
    for i in range(len(left)):
        sum += left[i] * right[i]
        print(sum)
    return sum

assert manual_dot_product(a, b) == 2  # 2
assert manual_dot_product(a, c) == 5  # 5
assert manual_dot_product(c, b) == 0  # 0

# use torch functions for manual dot product (not torch.dot() though)

assert a.multiply(b).sum() == 2
assert a.multiply(c).sum() == 5
assert c.multiply(b).sum() == 0

# OR:

assert (a * b).sum() == 2
assert (a * c).sum() == 5
assert (c * b).sum() == 0

# %% 1D

# matrix multiplication
quantities = tensor([1, 8, 4.0])
prices_per_unit = tensor([1, 0.25, 2])
quantities.matmul(prices_per_unit)

# reflect on how this works!? is matmul == batched dot() product?
# length of vector == a.dot(a).sqrt() # with self!

kata.helpers.DEBUG = True
assert_close(a.dot(a), tensor(1 * 1 + 2 * 2 + 4 * 4))

# %% 2D

quantities = tensor([1, 8, 4.0])
prices_per_unit = tensor([1, 0.25, 2])
quantities.matmul(prices_per_unit)
