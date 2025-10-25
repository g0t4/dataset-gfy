from typing import Any

import torch
from torch import tensor, Tensor

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

# reflect on how this works!? is matmul == batched/permuted dot() product?
#  - permuted in that it is going to be every vector (right side) dotted with every vector (left side)

kata.helpers.DEBUG = True
assert_close(a.dot(a), tensor(1 * 1 + 2 * 2 + 4 * 4))

# %% 1-D vector dotted w/ self computes the square of its magnitude

vector = tensor([3, 4])  # magnitude is gonna be 5
vector.dot(vector).sqrt()  # 5!
# simply put, sqrt(a*a + b*b + c*c ... + n*n) == pythagorean theorem
# length of vector == a.dot(a).sqrt() # with self!

vector.matmul(vector).sqrt()  # same result, 1Dx1D matmul == dot product...

# %% 2D matrix matmul

vectors = tensor([
    [3, 4],
    [6, 8],
])
vectors.T
vectors.matmul(vectors.T)  # in 2-D the diagonal (upper left, to lower right) represents the dot product of each vector w/ itself

# %% multiple order matrix

order_quantities = tensor([
    [1, 8, 4.0],  # 1x item 0, 8x item 1, 4x item2
    [2, 0, 0],  # 2x item 0
])
prices_per_unit = tensor([1, 0.25, 2])
order_costs = order_quantities.matmul(prices_per_unit)
assert_close(
    order_costs,
    [11, 2],  # order 0 is $11, order 1 is $2
)

# %% squeeze (aka expand_dim)

kata.helpers.DEBUG = True
order = tensor([1, 8, 4])
assert_close(order.shape, [3])

order.unsqueeze(0)  # adds outermost dimension
order.unsqueeze(1)  # adds innermost dimension (in this case there's just one dimension to start)
order.unsqueeze(-1)  # IIRC -1 is innermost always
assert_close(order.unsqueeze(1), order.unsqueeze(-1))
assert_close(order.unsqueeze(0).shape, [1, 3])
assert_close(order.unsqueeze(1).shape, [3, 1])  # 3 rows of 1 item each

# %% TODO dim= is an area that confuses me, actually the conventions around dimensions is what I haven't internalized

# %% TODO projections - start along axes and then move to vectors that point wherever

# %%

values = tensor([
    [2, 1, 3],
    [0, 1, -1],
], dtype=torch.float16)
# mean =>
# add up each dimension's value, then divide by # values (tensors)

values.mean(dim=0)  # [1,1,1]
values.mean(dim=1)  # 6/3=2, 0/3 = 0 => # [2, 0]
values.mean()  # across all values => 2+1+3+0+1-1 => 6 / 6 values => 1

# %%
def summarize_tensor(tensor: torch.Tensor):
    return

one_d = tensor([1, 2, 3.0])
one_d.mean()

two_d = tensor([
    [1, 2, 3.0],  # mean=2
    [3, 4, 5],  # mean=4
])  # overall mean=3
two_d[0].mean()  # y = 0
two_d[1].mean()  # y = 1
two_d.mean()  # 3
two_d.mean(dim=0)  # (2,3,4) - columns
two_d.mean(dim=1)  # rows
two_d.mean(dim=-1)  # rows too (2,4) # by convention -1 == last == innermost
two_d.mean(dim=-2)  # columns too (2,3,4)
two_d[:1]  #first row (0 to 1-not inclusive ==> 0 only)

# NO:
two_d[0] == two_d[:1]
two_d[0]
two_d[:0]

# %%

