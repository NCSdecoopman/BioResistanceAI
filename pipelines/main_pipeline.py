import argparse
from pipelines import prepare_data, model_data


def run_prepare_data(params_path: str):
    prepare_data.main(params_path=params_path)


def run_model_data(params_path: str, training_config_path: str, models_path: str):
    model_data.main(
        params_path=params_path,
        training_config_path=training_config_path,
        models_path=models_path
    )


def main(
    params_path: str = "config/params.yaml",
    training_config_path: str = "config/train_config.yaml",
    models_path: str = "config/models.yaml"
):
    print("Étape 1 : Préparation des données")
    run_prepare_data(params_path)

    print("\nÉtape 2 : Entraînement des modèles")
    run_model_data(params_path, training_config_path, models_path)

    print("\nPipeline terminée avec succès.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline complète (prétraitement + entraînement ML).")
    parser.add_argument("--params", default="config/params.yaml", help="Chemin vers le fichier params.yaml")
    parser.add_argument("--train_config", default="config/train_config.yaml", help="Chemin vers le fichier de configuration d'entraînement")
    parser.add_argument("--models", default="config/models.yaml", help="Chemin vers le fichier de modèles")

    args = parser.parse_args()

    main(
        params_path=args.params,
        training_config_path=args.train_config,
        models_path=args.models
    )
