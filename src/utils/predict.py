# src/utils/predict.py

import numpy as np

def safe_predict(estimator, X, model_name):
    
    try:
        from xgboost import DMatrix
    except ImportError:
        DMatrix = None

    # Cas 1 : Prédiction Skorch (PyTorch) — nécessite un array float32 sans colonnes nommées
    if "Skorch" in model_name or hasattr(estimator, "module_"):
        if hasattr(X, "values"):
            X = X.values.astype("float32")
        else:
            X = X.astype("float32")
        return estimator.predict(X)

    # Cas 2 : Prédiction XGBoost (array float32, gestion GPU)
    if "XGB" in (model_name or "") and DMatrix is not None:
        try:
            arr = X.values if hasattr(X, "values") else X
            dmat = DMatrix(arr.astype(np.float32))
            dmat._device = 'cuda'
            return estimator.get_booster().predict(dmat)
        except Exception as e:
            print(f"[XGB fallback to CPU] {e}")
            return estimator.predict(arr.astype(np.float32))


    # Cas 3 : Par défaut (sklearn, LightGBM, etc) predict classique
    return estimator.predict(X)
