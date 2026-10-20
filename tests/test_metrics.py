from typing import Callable

from timellm.utils import metrics

import numpy as np

import pytest

@pytest.mark.parametrize(
    ("metric", "expected_answer"),
    [
        pytest.param(metrics.RSE, np.float32(1.5811388), id="RSE"),
        pytest.param(metrics.MAE, np.float32(1.0), id="MAE"),
        pytest.param(metrics.MSE, np.float32(1.6666666), id="MSE"),
        pytest.param(metrics.RMSE, np.float32(1.2909944), id="RMSE"),
        pytest.param(metrics.MAPE, np.float32(0.17777778), id="MAPE"),
        pytest.param(metrics.MSPE, np.float32(0.050370377), id="MSPE"),
    ]
)

def test_metrics(
    metric: Callable[[np.ndarray, np.ndarray], float],
    expected_answer: float,
) -> None:
    """
    Test metric response on simple data.

    true data is always [4, 5, 6], and the prediction is [4, 4, 4].
    """
    true = np.array([4, 5, 6], dtype = np.float32)
    pred = np.array([4, 4, 4], dtype = np.float32)
    assert metric(pred, true) == pytest.approx(expected_answer)


def test_metric():
    """Test metric function on simple data.

    The metric function simply returns a tuple of some of the functions tested above, in a specific order.
    """
    true = np.array([4, 5, 6], dtype = np.float32)
    pred = np.array([4, 4, 4], dtype = np.float32)
    assert metrics.metric(pred, true) == (np.float32(1.0), np.float32(1.6666666), np.float32(1.2909944), np.float32(0.17777778), np.float32(0.050370377))