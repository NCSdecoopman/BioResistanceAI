import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
import yaml

from src.data.load_data import load_raw_data
from src.data.preprocess import preprocess_data, save_interim_data

def test_preprocess_pipeline():
    # Chargement de la configuration
    config_path = Path("config/params.yaml")
    assert config_path.exists(), "Le fichier config/params.yaml est manquant."

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Chargement des données
    data = load_raw_data(config)
    X, y = preprocess_data(data)

    # Vérifications
    assert isinstance(X, pd.DataFrame), "X n'est pas un DataFrame"
    assert isinstance(y, pd.DataFrame), "y n'est pas un DataFrame"
    assert not X.empty, "Les features sont vides"
    assert not y.empty, "Les cibles sont vides"
    assert X.shape[0] == y.shape[0], "Mismatch X/y sur les lignes"

    print("test_preprocess_pipeline OK")

def test_save_interim_data():
    import tempfile
    import pandas as pd
    import numpy as np
    from pathlib import Path
    from src.data.preprocess import save_interim_data  # adapte si nécessaire

    # Données fictives
    X = pd.DataFrame(np.random.rand(10, 5), columns=[f"feat_{i}" for i in range(5)])
    y = pd.DataFrame({"target": np.random.randint(0, 2, 10)})

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        paths = {
            "interim": tmp_path,
            "X_name": "X_clean.parquet",
            "y_name": "y_clean.parquet"
        }

        save_interim_data(X, y, paths)

        # Chemins complets vers les fichiers sauvegardés
        x_file = tmp_path / "X_clean.parquet"
        y_file = tmp_path / "y_clean.parquet"

        assert x_file.exists(), "X_clean.parquet non créé"
        assert y_file.exists(), "y_clean.parquet non créé"


