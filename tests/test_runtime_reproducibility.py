import numpy as np
import pytest
from spd_connectome_benchmark.training import set_global_random_seed

torch = pytest.importorskip("torch")


def test_fixed_seed_repeats_numpy_and_torch_draws():
    set_global_random_seed(11)
    first_numpy = np.random.rand(4)
    first_torch = torch.rand(4)

    set_global_random_seed(11)
    assert np.array_equal(first_numpy, np.random.rand(4))
    assert torch.equal(first_torch, torch.rand(4))
