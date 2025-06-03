# src/models/model_selector.py

import yaml, importlib

def load_models(yaml_path: str):
    """Donne le dictionnaire contenant tous les modèles"""

    # Ouvre le fichier YAML qui contient la configuration des modèles
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    # Initialise un dictionnaire vide pour stocker les modèles
    models = {}

    # Parcourt chaque modèle défini dans la section "models" du YAML
    for name, info in config["models"].items():
        # Sépare le chemin complet de la classe en module et nom de classe
        module, class_name = info["class_path"].rsplit(".", 1)

        # Importe dynamiquement la classe à partir du module et du nom de classe
        cls = getattr(importlib.import_module(module), class_name)

        # Ajoute au dictionnaire sous le nom du modèle : la classe importée et ses paramètres associés
        models[name] = {"class": cls, "params": info["params"]}
    return models
