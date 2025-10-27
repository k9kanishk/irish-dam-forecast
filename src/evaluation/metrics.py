import numpy as np


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))


def mape(y_true, y_pred):
    e = np.abs((y_true - y_pred) / np.maximum(1e-6, np.abs(y_true)))
    return float(100*np.mean(e))
