import einops
from einops import rearrange
import torch
from torch import tensor
import subprocess, tempfile, cv2
import numpy as np

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

# %% add outer and inner! at same time

unsqueeze_both = order_quantities.unsqueeze(0).unsqueeze(-1)
rearrange_both = rearrange(order_quantities, "a -> 1 a 1")
assert_close(rearrange_both, unsqueeze_both)

# %% reduce (rearrange + reduce)

ten_nums = torch.arange(1, 11)
ten_nums
einops.reduce(ten_nums, "a -> 1", "sum")  # tensor([55])

einops.reduce(ten_nums, "a -> a", "sum")  # noop (no change)
einops.repeat(ten_nums, "a -> 3 a")  # 3x of ten_nums in a new outer dim

# %% torch.flatten

inputs = tensor([
    [
        [1, 2],
        [3, 4],
    ],
    [
        [5, 6],
        [7, 8],
    ],
])
inputs.flatten(start_dim=1, end_dim=2)  # flattens within each pair of pairs... so 1..4 and 5..8
inputs.flatten(start_dim=0, end_dim=1)  # array of pairs
inputs.flatten(start_dim=0, end_dim=2)  # flattens across all dimensions into list of numbers 1 to 8

# so IIUC () is for flattening
einops.rearrange(inputs, "a b c -> a (b c)")  # two lists of 4 nums (1..4 and 5..8)
einops.rearrange(inputs, "a b c -> (a b) c")  # array of pairs
einops.rearrange(inputs, "a b c -> (a b c)")  # flattens across all dims

# %%
import numpy as np
import os
from pathlib import Path

# test image from einops tutorial: https://einops.rocks/1-einops-basics/
test_images_npy  = Path(os.environ["WES_REPOS"]).joinpath("github/arogozhnikov/einops/docs/resources/test_images.npy")
ims = np.load(test_images_npy, allow_pickle=False)
first_image = ims[0]
first_image.dtype
first_image.shape
first_image[:1,:3,:] # first 3 pixels of first column (or row?)

# %%

def show_image(image: np.ndarray) -> None:
    import subprocess, tempfile, cv2
    import numpy as np
    path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    cv2.imwrite(path, image * 255)
    subprocess.run(["open", path])

show_image(first_image)


# %%

