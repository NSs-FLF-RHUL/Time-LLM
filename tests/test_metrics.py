from collections.abc import Callable

import numpy as np
import pytest

from timellm.utils import metrics


@pytest.mark.parametrize(
    ("metric", "expected_answer"),
    [
        pytest.param(metrics.RSE, np.float32(1.5811388), id="RSE"),
        pytest.param(metrics.MAE, np.float32(1.0), id="MAE"),
        pytest.param(metrics.MSE, np.float32(1.6666666), id="MSE"),
        pytest.param(metrics.RMSE, np.float32(1.2909944), id="RMSE"),
        pytest.param(metrics.MAPE, np.float32(0.17777778), id="MAPE"),
        pytest.param(metrics.MSPE, np.float32(0.050370377), id="MSPE"),
        pytest.param(
            metrics.metric,
            (1.0, 1.6666666, 1.2909944, 0.17777778, 0.050370377),
            id="metric",
        ),
    ],
)
def test_metrics(
    metric: Callable[[np.ndarray, np.ndarray], float],
    expected_answer: float,
) -> None:
    """
    Test metric response on simple data.

    true data is always [4, 5, 6], and the prediction is [4, 4, 4].
    """
    true = np.array([4, 5, 6], dtype=np.float32)
    pred = np.array([4, 4, 4], dtype=np.float32)
    metric_output = np.atleast_1d(metric(pred, true))
    expected_answer = np.atleast_1d(expected_answer)
    assert np.allclose(metric_output, expected_answer)
