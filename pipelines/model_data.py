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
import json

from contextlib import contextmanager
from tqdm import tqdm
import joblib

@contextmanager
def tqdm_joblib(tqdm_object):
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)
    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()



def save_partial_result(result_path, abx, abx_results):
    """Sauvegarde ou met à jour les résultats intermédiaires dans un fichier JSON."""
    if result_path.exists():
        with open(result_path, "r") as f:
            current_results = json.load(f)
    else:
        current_results = {}

    current_results[abx] = abx_results

    with open(result_path, "w") as f:
        json.dump(current_results, f, indent=2)



def run_for_abx(abx, X, y, feature_types, model_config, scaler_str, test_size, random_state, cv, scoring_func, n_jobs):
    y_abx = y[["strain_ids", abx]].dropna()
    X_abx = X.loc[y_abx.index]
    y_target = y_abx[abx]

    scaler = get_scaler(scaler_str)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(
        X_abx, y_target, test_size=test_size, random_state=random_state, scaler=scaler
    )

    abx_results = {}
    for model_name in tqdm(model_config.keys(), desc=f"{abx}", leave=False):
        model_info = model_config[model_name]
        start_time = time.time()

        params = model_info["params"]
        device = params.get("device", ["cpu"])
        if isinstance(device, str):
            device = [device]
        device_is_gpu = any("cuda" in str(d).lower() or "gpu" in str(d).lower() for d in device)
        model_n_jobs = 1 if device_is_gpu else n_jobs

        if model_info["class"] == "skorch.NeuralNetClassifier":
            module_path = params.get("module", [None])[0]
            if module_path:
                module_name, class_name = module_path.rsplit(".", 1)
                module = importlib.import_module(module_name)
                params["module"] = getattr(module, class_name)

        if model_name == "SkorchMLPClassifier":
            input_dim = X_train.shape[1]
            model_info["params"]["module__input_dim"] = [input_dim]

        clf = train_with_gridsearch(
            model_info["class"], model_info["params"],
            X_train, y_train, cv=cv, n_jobs=model_n_jobs, scoring=scoring_func
        )

        train_duration = time.time() - start_time
        y_pred = safe_predict(clf.best_estimator_, X_test, model_name)
        score = scoring_func._score_func(y_test, y_pred) if hasattr(scoring_func, "_score_func") else scoring_func(y_test, y_pred)

        contributions = compute_group_contributions(
            clf.best_estimator_, X_abx.copy(), y_target,
            scaler, feature_types, score, random_state
        )

        abx_results[model_name] = {
            scoring_func.__name__ if hasattr(scoring_func, "__name__") else "score": round(score, 3),
            "best_params": clf.best_params_,
            "contributions_%": contributions,
            "train_time_sec": round(train_duration, 2)
        }

        print(f"{abx} - {model_name} : recall={round(score, 3)} | durée={round(train_duration, 2)}s")

    # Sauvegarde intermédiaire
    save_partial_result(Path("results/partial_results.json"), abx, abx_results)

    return abx, abx_results

def main(params_path: str = "config/params.yaml",
         training_config_path: str = "config/train_config.yaml",
         models_path: str = "config/models.yaml"):

    config_paths = load_yaml(params_path)["paths"]
    training_config = load_yaml(training_config_path)["training"]
    model_config = load_models(models_path)

    interim_dir = Path(config_paths["interim"])
    X_path = interim_dir / config_paths["X_name"]
    y_path = interim_dir / config_paths["y_name"]
    result_path = Path(config_paths["result"]) / "all_model_results.json"

    X, y = load_clean_data(X_path, y_path)
    feature_types = get_feature_groups(X)

    test_size = training_config.get("test_size", 0.2)
    random_state = training_config.get("random_state", 42)
    cv = training_config.get("cv", 5)
    n_jobs = training_config.get("n_jobs", -1)
    scoring_key = training_config.get("scoring", "recall_macro")
    scoring_func = SCORERS.get(scoring_key, scoring_key)
    scaler_str = training_config.get("scaler", "sklearn.preprocessing.StandardScaler")

    # Parallélisation sur les antibiotiques
    abx_list = list(y.columns[1:])
    with tqdm_joblib(tqdm(desc="Antibiotiques", total=len(abx_list))) as progress_bar:
        results = joblib.Parallel(n_jobs=5)(
            joblib.delayed(run_for_abx)(abx, X, y, feature_types, model_config, scaler_str,
                                test_size, random_state, cv, scoring_func, n_jobs)
            for abx in abx_list
        )


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