# pipelines/main
# Le pipeline charge ses paramètres depuis des fichiers YAML, prépare les données, puis entraîne et évalue en parallèle plusieurs modèles de machine learning pour chaque antibiotique, 
# tout en sauvegardant les résultats et leur progression.

import argparse
import importlib
import time
from pathlib import Path

from src.data.load_data import load_clean_data
from src.data.split_scale import split_and_scale, get_scaler
from src.features.feature_groups import get_feature_groups
from src.utils.predict import safe_predict
from src.models.model_selector import load_models
from src.models.train_model import train_with_gridsearch
from src.models.contributions import compute_group_contributions
from src.evaluation.metrics import SCORERS
from src.evaluation.reporting import save_results
from src.data.config import load_yaml
from src.utils.json import make_json_serializable
import json

from contextlib import contextmanager
from tqdm import tqdm
import joblib

# Contexte permettant d'afficher une barre de progression tqdm lors de l'utilisation de joblib.Parallel.
@contextmanager
def tqdm_joblib(tqdm_object):
    # On définit une sous-classe du callback utilisé par joblib à chaque lot (batch) terminé
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            # À chaque batch terminé, on met à jour la barre de progression de tqdm
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    # On sauvegarde l'ancienne classe de callback pour la restaurer plus tard
    old_callback = joblib.parallel.BatchCompletionCallBack
    # On remplace le callback par notre version custom qui met à jour tqdm
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback

    try:
        # On fournit tqdm_object au contexte (utile si on veut l'utiliser dans le bloc with)
        yield tqdm_object
    finally:
        # On restaure le callback d'origine pour ne pas impacter les autres utilisations de joblib
        joblib.parallel.BatchCompletionCallBack = old_callback
        # On ferme proprement la barre de progression
        tqdm_object.close()



def save_partial_result(result_path, abx, abx_results):
    """Sauvegarde ou met à jour les résultats intermédiaires dans un fichier JSON."""
    if result_path.exists():
        with open(result_path, "r") as f:
            current_results = json.load(f)
    else:
        current_results = {}

    if abx not in current_results:
        current_results[abx] = {}

    # Ajouter ou mettre à jour les modèles un par un
    current_results[abx].update(abx_results)

    with open(result_path, "w") as f:
        json.dump(current_results, f, indent=2)




def resolve_string_to_class(val):
    # Résout dynamiquement une chaîne de caractères en une classe Python.
    # >>> resolve_string_to_class("collections.Counter")
    # <class 'collections.Counter'>
    # >>> resolve_string_to_class(list)
    # <class 'list'>
    # >>> resolve_string_to_class("Counter")   # Sans point, retourne la chaîne elle-même
    # 'Counter'
    if isinstance(val, str) and "." in val:
        module_name, class_name = val.rsplit(".", 1)
        return getattr(importlib.import_module(module_name), class_name)
    return val



def run_for_abx(abx, X, y, feature_types, model_config, scaler_str, test_size, random_state, cv, scoring_func, n_jobs):
    """
    Entraîne et évalue tous les modèles spécifiés dans model_config pour un antibiotique donné.
    Gère la persistance des résultats partiels pour reprendre un calcul interrompu.

    Paramètres
    ----------
    abx : str
        Nom de l'antibiotique cible à prédire.
    X : DataFrame
        Table des features (observations × variables).
    y : DataFrame
        Table des étiquettes, avec colonne "strain_ids" et une colonne par antibiotique.
    feature_types : dict
        Dict qui indique l'origine ou le groupe de chaque feature (pour importance des groupes).
    model_config : dict
        Configuration des modèles à entraîner (nom → paramètres).
    scaler_str : str
        Nom du scaler à utiliser.
    test_size : float
        Taille du test set (ex : 0.2 pour 20%).
    random_state : int
        Graine de hasard.
    cv : int
        Nombre de folds pour la cross-validation.
    scoring_func : callable
        Fonction de scoring (ex : recall_score).
    n_jobs : int
        Nb de threads à utiliser (sauf si GPU, alors 1).
    """

    # 1. Sélectionne les échantillons non NaN pour cet antibiotique
    y_abx = y[["strain_ids", abx]].dropna()
    X_abx = X.loc[y_abx.index]
    y_target = y_abx[abx]

    # 2. Scaling et split
    scaler = get_scaler(scaler_str)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(
        X_abx, y_target, 
        test_size=test_size, 
        random_state=random_state, 
        scaler=scaler
    )

    abx_results = {}

    # 3. Chargement des résultats partiels (pour reprendre sans tout recalculer)
    partial_result_path = Path("results/partial_results.json")
    if partial_result_path.exists():
        with open(partial_result_path, "r") as f:
            partial_results = json.load(f)
    else:
        partial_result_path.parent.mkdir(parents=True, exist_ok=True)
        partial_results = {}
        with open(partial_result_path, "w") as f:
            json.dump(partial_results, f, indent=2)
    
    # 4. Pour chaque modèle de la config, entraînement et évaluation
    for model_name in tqdm(model_config.keys(), desc=f"{abx}", leave=False):
        # Skip si déjà traité (checkpoint)
        if abx in partial_results and model_name in partial_results[abx]:
            print(f"Skip {abx} - {model_name} (already computed)")
            continue

        model_info = model_config[model_name]
        start_time = time.time()

        params = model_info["params"]

        # Résolution dynamique des strings vers classes/fonctions pour criterion, optimizer, module
        for key in ["criterion", "optimizer", "module"]:
            if key in params:
                if isinstance(params[key], list):
                    params[key] = [resolve_string_to_class(v) for v in params[key]]
                else:
                    params[key] = resolve_string_to_class(params[key])

        # Gestion CPU/GPU (sklearn ne supporte pas GPU, torch/Skorch oui)
        device = params.get("device", ["cpu"])
        if isinstance(device, str):
            device = [device]
        device_is_gpu = any("cuda" in str(d).lower() or "gpu" in str(d).lower() for d in device)
        model_n_jobs = 1 if device_is_gpu else n_jobs

        # Convertir le nom de la classe string en classe réelle
        module_path = model_info["class"]
        if isinstance(module_path, str):
            module_name, class_name = module_path.rsplit(".", 1)
            model_class = getattr(importlib.import_module(module_name), class_name)
        else:
            model_class = model_info["class"]  # déjà une classe

        # Charger dynamiquement le module Skorch si besoin
        if issubclass(model_class, importlib.import_module("skorch.classifier").NeuralNetClassifier):
            if "module" in params:
                if isinstance(params["module"], list):
                    params["module"] = [resolve_string_to_class(v) for v in params["module"]]
                else:
                    params["module"] = resolve_string_to_class(params["module"])

        # Conversion dynamique du nom d'optimizer si c’est une string
        if isinstance(params.get("optimizer"), list):
            opt_str = params["optimizer"][0]
            if isinstance(opt_str, str) and "." in opt_str:
                module_name, class_name = opt_str.rsplit(".", 1)
                optimizer_module = importlib.import_module(module_name)
                params["optimizer"] = [getattr(optimizer_module, class_name)]
        
        # Injecter input_dim pour les modèles MLP Skorch
        if model_name == "SkorchMLPClassifier":
            input_dim = X_train.shape[1]
            model_info["params"]["module__input_dim"] = [input_dim]


        X_tr, X_te = X_train, X_test

       # Entraînement avec GridSearchCV
        clf = train_with_gridsearch(
            model_class,
            model_info["params"],
            X_tr, y_train, 
            cv=cv, n_jobs=model_n_jobs, 
            scoring=scoring_func
        )

        train_duration = time.time() - start_time

        # Prédiction sur test set
        y_pred = safe_predict(clf.best_estimator_, X_te, model_name)

        # Calcul du score
        score = (scoring_func._score_func(y_test, y_pred) 
                if hasattr(scoring_func, "_score_func") 
                else scoring_func(y_test, y_pred))

        # Importance des groupes de variables (GPA, SNPs, etc.)
        contributions = compute_group_contributions(
            clf.best_estimator_, X_abx.copy(), y_target,
            scaler, feature_types, score, random_state
        )

        # Construction du résultat sérialisable
        result_dict = {
            scoring_func.__name__ if hasattr(scoring_func, "__name__") else "score": round(score, 3),
            "best_params": make_json_serializable(clf.best_params_),
            "contributions_%": contributions,
            "train_time_sec": round(train_duration, 2)
        }

        abx_results[model_name] = result_dict

        print(f"{abx} - {model_name} : recall={round(score, 3)} | durée={round(train_duration, 2)}s")

        # Sauvegarde du résultat pour ce modèle/abx uniquement
        save_partial_result(Path("results/partial_results.json"), abx, {model_name: result_dict})

    return abx, abx_results


def main(params_path: str = "config/params.yaml",
         training_config_path: str = "config/train_config.yaml",
         models_path: str = "config/models.yaml"):

    # Chargement des configs (données, modèles, entraînement)
    config_paths = load_yaml(params_path)["paths"]
    training_config = load_yaml(training_config_path)["training"]
    model_config = load_models(models_path)

    # Construction des chemins des fichiers de travail à partir de la config
    interim_dir = Path(config_paths["interim"])
    X_path = interim_dir / config_paths["X_name"]
    y_path = interim_dir / config_paths["y_name"]
    result_path = Path(config_paths["result"]) / "all_model_results.json"

    # Chargement des données features (X) et cibles (y) pré-nettoyées
    X, y = load_clean_data(X_path, y_path)

    # Extraction des groupes de features (pour l’analyse des importances par groupe)
    feature_types = get_feature_groups(X)

    # Chargement des paramètres globaux d'entraînement
    test_size = training_config.get("test_size", 0.2)
    random_state = training_config.get("random_state", 42)
    cv = training_config.get("cv", 5)
    n_jobs = training_config.get("n_jobs", -1)
    scoring_key = training_config.get("scoring", "recall_macro")
    scoring_func = SCORERS.get(scoring_key, scoring_key)
    scaler_str = training_config.get("scaler", "sklearn.preprocessing.StandardScaler")

    # Constitution de la liste des antibiotiques à traiter (toutes les colonnes sauf strain_ids)
    abx_list = list(y.columns[1:])

    # Boucle parallèle sur chaque antibiotique avec suivi de progression (tqdm)
    with tqdm_joblib(tqdm(desc="Antibiotiques", total=len(abx_list))) as progress_bar:
        results = joblib.Parallel(n_jobs=5)(
            joblib.delayed(run_for_abx)(abx, X, y, feature_types, model_config, scaler_str,
                                test_size, random_state, cv, scoring_func, n_jobs)
            for abx in abx_list
        )

    # Construction du dictionnaire global des résultats et sauvegarde des résultats globaux dans un fichier .json
    all_results = {abx: res for abx, res in results}
    save_results(all_results, result_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline d'entraînement ML avec configuration YAML.")
    parser.add_argument("--params", default="config/params.yaml", help="Chemin vers le fichier params.yaml")
    parser.add_argument("--train_config", default="config/train_config.yaml", help="Chemin vers le fichier de configuration d'entraînement")
    parser.add_argument("--models", default="config/models.yaml", help="Chemin vers le fichier de modèles")

    args = parser.parse_args()

    main(
        params_path=args.params,
        training_config_path=args.train_config,
        models_path=args.models
    )