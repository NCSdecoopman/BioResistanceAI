# pipelines/prepare_data.py
import argparse
from pathlib import Path
import subprocess
import yaml

from src.data.load_data import load_raw_data
from src.data.preprocess import preprocess_data, save_interim_data

def main(
        params_path: str = "config/params.yaml"
    ):
    with open(params_path, "r") as f:
        config = yaml.safe_load(f)

    paths = config.get("paths", {})

    print("Chargement des fichiers...")
    data = load_raw_data(config)

    print("Prétraitement et fusion des données...")
    X, y = preprocess_data(data)

    print("Sauvegarde des données nettoyées...")
    save_interim_data(X, y, paths)
    print(f"Données explicates prêtes : {X.shape[0]} lignes, {X.shape[1]} variables explicatives")
    print(X.head())

    print(f"\nDonnées à expliquer prêtes : {y.shape[0]} lignes, {y.shape[1]} variables :")
    print(y.head())    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de préparation des données.")
    parser.add_argument("--params", default="config/params.yaml", help="Chemin vers le fichier params.yaml")

    args = parser.parse_args()

    main(
        params_path=args.params
    )
