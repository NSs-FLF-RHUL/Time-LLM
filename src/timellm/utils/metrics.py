# ruff: noqa: N802

"""metrics computes metrics between arrays of true and predicted values."""

import numpy as np


def RSE(pred: np.ndarray, true: np.ndarray) -> np.float16 | np.float32 | np.float64:
    """Relative Squared Error."""
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(
        np.sum((true - true.mean()) ** 2)
    )


def CORR(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """
    Compute a Correlation Coefficient.

    The formula looks like the formula for the Pearson correlation coefficient,
    but the denominator is different.
    """
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred: np.ndarray, true: np.ndarray) -> np.float16 | np.float32 | np.float64:
    """Mean Absolute Error."""
    return np.mean(np.abs(pred - true))


def MSE(pred: np.ndarray, true: np.ndarray) -> np.float16 | np.float32 | np.float64:
    """Mean Squared Error."""
    return np.mean((pred - true) ** 2)


def RMSE(pred: np.ndarray, true: np.ndarray) -> np.float16 | np.float32 | np.float64:
    """Root Mean Squared Error."""
    return np.sqrt(MSE(pred, true))


def MAPE(pred: np.ndarray, true: np.ndarray) -> np.float16 | np.float32 | np.float64:
    """Mean Absolute Relative Error."""
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred: np.ndarray, true: np.ndarray) -> np.float16 | np.float32 | np.float64:
    """Mean Squared Relative Error."""
    return np.mean(np.square((pred - true) / true))


def metric(pred: np.ndarray, true: np.ndarray) -> tuple:
    """Compute and returns some standard metrics between true and predicted values."""
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe
