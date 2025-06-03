# src/data/load_data.py
import pandas as pd
import numpy as np
from pathlib import Path

def load_raw_data(config: dict) -> dict:
    """
    Charge les données converties (parquet et npy) en utilisant les chemins spécifiés dans le fichier de config.
    """
    paths_cfg = config["paths"]
    data_dir = Path(paths_cfg["converted_dir"])

    pheno_path = data_dir / paths_cfg["pheno"]
    X_gpa_path = data_dir / paths_cfg["X_gpa"]
    X_snps_path = data_dir / paths_cfg["X_snps"]
    X_genexp_path = data_dir / paths_cfg["X_genexp"]

    if not (pheno_path.exists() and X_gpa_path.exists() and X_snps_path.exists() and X_genexp_path.exists()):
        raise FileNotFoundError("Un ou plusieurs fichiers de données converties sont manquants.")

    return {
        "pheno": pd.read_parquet(pheno_path),
        "X_gpa": np.load(X_gpa_path),
        "X_snps": np.load(X_snps_path),
        "X_genexp": np.load(X_genexp_path),
    }

def load_clean_data(path_X: str, path_y: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = pd.read_parquet(path_X)
    y = pd.read_parquet(path_y)
    return X, y