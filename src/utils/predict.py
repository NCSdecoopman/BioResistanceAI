import cupy as cp

def safe_predict(estimator, X, model_name):
    # Cas XGBoost → conversion explicite en DMatrix avec bon device
    if "XGB" in model_name:
        try:
            from xgboost import DMatrix
            dmatrix = DMatrix(cp.asarray(X.values))
            return estimator.get_booster().predict(dmatrix)
        except Exception:
            pass  # fallback CPU si problème

    # Cas Skorch ou LGBM avec GPU (si données en CuPy ne posent pas de souci)
    if "Skorch" in model_name or "LGBM" in model_name:
        try:
            X_gpu = cp.asarray(X.values)
            return estimator.predict(X_gpu)
        except Exception:
            pass  # fallback CPU

    # Cas par défaut (CPU ou modèle classique)
    return estimator.predict(X)
