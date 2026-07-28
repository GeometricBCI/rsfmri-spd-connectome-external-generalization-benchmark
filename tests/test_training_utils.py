import pytest

from spd_connectome_benchmark.training import ensure_nonempty_training_batches


def test_zero_batch_training_fails_clearly_without_changing_drop_last():
    with pytest.raises(ValueError, match="zero batches"):
        ensure_nonempty_training_batches(10, 32, drop_last=True)
    assert ensure_nonempty_training_batches(10, 32, drop_last=False) == 1
