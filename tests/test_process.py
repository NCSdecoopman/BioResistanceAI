import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import recall_score
from pipelines.model_data import main

@pytest.fixture
def mock_config(tmp_path, mocker):
    # YAML simulé
    config_paths = {
        "paths": {
            "interim": str(tmp_path / "data" / "interim"),
            "X_name": "X_clean.parquet",
            "y_name": "y_clean.parquet",
            "result": str(tmp_path / "results"),
        }
    }

    training_config = {
        "training": {
            "test_size": 0.2,
            "random_state": 42,
            "cv": 3,
            "n_jobs": 1,
            "scoring": "recall_macro",
            "scaler": "sklearn.preprocessing.StandardScaler"
        }
    }

    models_config = {
        "DummyModel": {
            "class": "sklearn.dummy.DummyClassifier",
            "params": {"strategy": ["most_frequent"]}
        }
    }

    # Création X avec au moins une colonne 'genexp_'
    columns = ["gpa_0", "gpa_1", "snps_0", "snps_1", "genexp_0", "genexp_1"]
    X = pd.DataFrame(np.random.rand(10, len(columns)), columns=columns)
    y = pd.DataFrame({
        "strain_ids": list(range(10)),
        "abx1": np.random.choice([0, 1], size=10)
    })

    # Patch les fonctions YAML
    mocker.patch("pipelines.model_data.load_yaml", side_effect=[config_paths, training_config])
    mocker.patch("pipelines.model_data.load_models", return_value=models_config)

    # Patch les données
    mocker.patch("pipelines.model_data.load_clean_data", return_value=(X, y))
    mocker.patch("pipelines.model_data.get_feature_groups", return_value={
        "gpa_": ["gpa_0", "gpa_1"],
        "snps_": ["snps_0", "snps_1"],
        "genexp_": ["genexp_0", "genexp_1"]
    })

    # Patch scalers et split
    scaler = mocker.MagicMock()
    mocker.patch("pipelines.model_data.get_scaler", return_value=scaler)
    mocker.patch("pipelines.model_data.split_and_scale", return_value=(X, X, y["abx1"], y["abx1"], scaler))

    # Patch GridSearch
    mock_best_estimator = mocker.MagicMock()
    mock_best_estimator.predict.return_value = y["abx1"]
    mock_clf = mocker.MagicMock()
    mock_clf.best_estimator_ = mock_best_estimator
    mock_clf.best_params_ = {"strategy": "most_frequent"}
    mocker.patch("pipelines.model_data.train_with_gridsearch", return_value=mock_clf)
    mocker.patch("pipelines.model_data.Path.exists", return_value=False)

    # Patch contributions
    mocker.patch("pipelines.model_data.compute_group_contributions", return_value={
        "gpa_": 40.0,
        "snps_": 30.0,
        "genexp_": 30.0
    })

    # Patch sauvegarde
    mock_save = mocker.patch("pipelines.model_data.save_results")

    return tmp_path, mock_save

def test_main_pipeline_runs(mock_config):
    tmp_path, mock_save = mock_config

    main(
        params_path="params.yaml",
        training_config_path="train_config.yaml",
        models_path="models.yaml"
    )

    # Vérifie que les résultats sont sauvegardés
    mock_save.assert_called_once()
    results = mock_save.call_args[0][0]

    assert "abx1" in results
    assert "DummyModel" in results["abx1"]

    # Cherche la bonne clé de score (tolère "score" ou "recall_macro")
    found = False
    for key in ("recall_macro", "score"):
        if key in results["abx1"]["DummyModel"]:
            assert isinstance(results["abx1"]["DummyModel"][key], float)
            assert 0.0 <= results["abx1"]["DummyModel"][key] <= 1.0
            found = True
            break
    assert found, "Aucune clé de score trouvée ('recall_macro' ou 'score')"

