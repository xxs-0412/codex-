import numpy as np
import joblib

_models = joblib.load("gpr_models.pkl")
_scalers = joblib.load("gpr_scalers.pkl")
_bounds = joblib.load("stage_bounds.pkl")
q1, q2 = _bounds['q1'], _bounds['q2']

def predict_k(F, v, r, t, return_std=False):
    if t <= q1:
        s = 0
    elif t <= q2:
        s = 1
    else:
        s = 2

    X = np.array([[F, v, r, t]])
    X = _scalers[s].transform(X)

    if return_std:
        log_k, std = _models[s].predict(X, return_std=True)
        return float(np.exp(log_k[0])), float(std[0])
    else:
        log_k = _models[s].predict(X)
        return float(np.exp(log_k[0]))