# src/data/split_scale.py

from sklearn.model_selection import train_test_split
import importlib
import pandas as pd

def get_scaler(scaler_path: str):
    """Charge dynamiquement un scaler à partir d’un chemin complet, ex : sklearn.preprocessing.StandardScaler"""
    module, class_name = scaler_path.rsplit(".", 1)
    return getattr(importlib.import_module(module), class_name)()

def split_and_scale(X, y, test_size, random_state, scaler):
    """
    Effectue un split train/test + applique le scaler uniquement sur les colonnes `genexp_`.
    
    Paramètres :
        - X (DataFrame) : features combinées avec préfixes `gpa_`, `snp_`, `genexp_`
        - y : cible
        - test_size, random_state : contrôle du split
        - scaler : instance sklearn (StandardScaler, etc.)
    
    Retourne :
        X_train_scaled, X_test_scaled, y_train, y_test, scaler (ajusté)
    """
    # 1) train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state
    )

    # 2) scale only genexp_
    genexp_cols = [c for c in X.columns if c.startswith("genexp_")]
    other_cols  = [c for c in X.columns if not c.startswith("genexp_")]

    X_tr_gen = scaler.fit_transform(X_train[genexp_cols])
    X_te_gen = scaler.transform(X_test[genexp_cols])

    # 3) reconstitue DataFrame (juste pour garder la logique)
    X_tr_df = pd.concat([
        X_train[other_cols].reset_index(drop=True),
        pd.DataFrame(X_tr_gen, columns=genexp_cols)
    ], axis=1).astype("float32")

    X_te_df = pd.concat([
        X_test[other_cols].reset_index(drop=True),
        pd.DataFrame(X_te_gen, columns=genexp_cols)
    ], axis=1).astype("float32")

    # Les y doivent rester Series
    y_tr_arr = y_train
    y_te_arr = y_test

    return X_tr_df, X_te_df, y_tr_arr, y_te_arr, scaler
