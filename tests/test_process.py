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

    # Patch les fonctions YAML
    mocker.patch("pipelines.model_data.load_yaml", side_effect=[config_paths, training_config])
    mocker.patch("pipelines.model_data.load_models", return_value=models_config)

    # Données fictives
    X = pd.DataFrame(np.random.rand(10, 6), columns=[f"feat_{i}" for i in range(6)])
    y = pd.DataFrame({
        "strain_ids": list(range(10)),
        "abx1": np.random.choice([0, 1], size=10)
    })

    # Patch les données
    mocker.patch("pipelines.model_data.load_clean_data", return_value=(X, y))
    mocker.patch("pipelines.model_data.get_feature_groups", return_value={
        "gpa_": list(X.columns[:2]),
        "snps_": list(X.columns[2:4]),
        "genexp_": list(X.columns[4:])
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

    # Fonction de scoring
    mocker.patch.dict("pipelines.model_data.SCORERS", {"recall_macro": recall_score})

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
    assert isinstance(results["abx1"]["DummyModel"]["recall_macro"], float)
    assert 0.0 <= results["abx1"]["DummyModel"]["recall_macro"] <= 1.0
