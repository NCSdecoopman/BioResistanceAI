# src/data/preprocess.py

import pandas as pd
from pathlib import Path

def preprocess_data(data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Nettoie et aligne les données brutes du dataset.

    Retourne :
        X (DataFrame): variables explicatives fusionnées, sans NaN
        y (DataFrame): cibles (phénotypes), index synchrone avec X
    """
    pheno = data["pheno"]
    X_gpa    = pd.DataFrame(data["X_gpa"],    index=pheno.index)
    X_snps   = pd.DataFrame(data["X_snps"],   index=pheno.index)
    X_genexp = pd.DataFrame(data["X_genexp"], index=pheno.index)

    # Préfixes
    X_gpa.columns    = [f"gpa_{i}"    for i in range(X_gpa.shape[1])]
    X_snps.columns   = [f"snp_{i}"    for i in range(X_snps.shape[1])]
    X_genexp.columns = [f"genexp_{i}" for i in range(X_genexp.shape[1])]

    # Supprimer colonnes constantes
    for X in (X_gpa, X_snps, X_genexp):
        nunique = X.nunique()
        X.drop(columns=nunique[nunique <= 1].index, inplace=True)

    # Fusion horizontale
    X_full = pd.concat([X_gpa, X_snps, X_genexp], axis=1)

    # —> SUPPRESSION DES LIGNES CONTENANT AU MOINS UN NaN
    mask = ~X_full.isna().any(axis=1)
    X_full = X_full.loc[mask]
    pheno  = pheno.loc[mask]

    return X_full, pheno


def save_interim_data(X: pd.DataFrame, y: pd.DataFrame, paths: dict):
    """Sauvegarde les données nettoyées dans les chemins définis dans params.yaml"""
    interim_dir = Path(paths.get("interim", "data/interim"))
    x_path      = interim_dir / paths.get("X_name", "X_clean.parquet")
    y_path      = interim_dir / paths.get("y_name", "y_clean.parquet")

    interim_dir.mkdir(parents=True, exist_ok=True)
    X.to_parquet(x_path)
    y.to_parquet(y_path)
