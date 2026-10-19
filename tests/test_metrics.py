from timellm.utils import metrics

import numpy as np


def test_rse():
    true = np.array([4, 5, 6], dtype = np.float32)
    pred = np.array([4, 4, 4], dtype = np.float32)
    assert metrics.RSE(pred, true) == np.float32(1.5811388)


def test_mae():
    true = np.array([4, 5, 6], dtype = np.float32)
    pred = np.array([4, 4, 4], dtype = np.float32)
    assert metrics.MAE(pred, true) == np.float32(1.0)


def test_mse():
    true = np.array([4, 5, 6], dtype = np.float32)
    pred = np.array([4, 4, 4], dtype = np.float32)
    assert metrics.MSE(pred, true) == np.float32(1.6666666)


def test_rmse():
    true = np.array([4, 5, 6], dtype = np.float32)
    pred = np.array([4, 4, 4], dtype = np.float32)
    assert metrics.RMSE(pred, true) == np.float32(1.2909944)


def test_mape():
    true = np.array([4, 5, 6], dtype = np.float32)
    pred = np.array([4, 4, 4], dtype = np.float32)
    assert metrics.MAPE(pred, true) == np.float32(0.17777778)


def test_mspe():
    true = np.array([4, 5, 6], dtype = np.float32)
    pred = np.array([4, 4, 4], dtype = np.float32)
    assert metrics.MSPE(pred, true) == np.float32(0.050370377)


def test_metric():
    true = np.array([4, 5, 6], dtype = np.float32)
    pred = np.array([4, 4, 4], dtype = np.float32)
    assert metrics.metric(pred, true) == (np.float32(1.0), np.float32(1.6666666), np.float32(1.2909944), np.float32(0.17777778), np.float32(0.050370377))