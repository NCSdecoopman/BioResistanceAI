def safe_predict(estimator, X, model_name):
    import numpy as np
    try:
        from xgboost import DMatrix
    except ImportError:
        DMatrix = None
    import skorch

    # Pour Skorch : array float32 sans nom de colonnes
    if "Skorch" in model_name or hasattr(estimator, "module_"):
        if hasattr(X, "values"):
            X = X.values.astype("float32")
        else:
            X = X.astype("float32")
        return estimator.predict(X)

    # Pour XGBoost : gérer GPU + array float32 si demandé
    if "XGB" in (model_name or "") and DMatrix is not None:
        try:
            arr = X.values if hasattr(X, "values") else X
            dmat = DMatrix(arr.astype(np.float32))
            dmat._device = 'cuda'
            return estimator.get_booster().predict(dmat)
        except Exception as e:
            print(f"[XGB fallback to CPU] {e}")
            return estimator.predict(arr.astype(np.float32))


    # Pour tout le reste (sklearn, LGBM, RF, SVM…)
    return estimator.predict(X)
