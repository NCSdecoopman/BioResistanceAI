def safe_predict(estimator, X, model_name):
    try:
        import cupy as cp
        if any(x in model_name for x in ["XGB", "Skorch", "LGBM"]):
            X_gpu = cp.asarray(X.values)
            return estimator.predict(X_gpu)
        else:
            return estimator.predict(X)
    except Exception:
        return estimator.predict(X)
