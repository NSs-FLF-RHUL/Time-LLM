import warnings

import numpy as np
import pytest

from timellm.utils import losses


def test_divide_no_nan():
    a = np.array([1, 2, np.nan, 4, 5, np.inf], dtype=np.float32)
    b = np.array([0, 2, 3, np.nan, 5, 6], dtype=np.float32)
    with pytest.warns() as record:
        warnings.warn("divide by zero encountered in divide", RuntimeWarning)
        np.testing.assert_array_equal(
            losses.divide_no_nan(a, b), np.array([0.0, 1, 0.0, 0.0, 1, 0.0])
        )
    assert len(record) == 2
    assert str(record[1].message) == "divide by zero encountered in divide"
