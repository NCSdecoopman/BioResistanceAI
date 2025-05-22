# BioResistanceAI

Prédiction de la résistance bactérienne aux antibiotiques à partir de données **génomiques**, **mutations SNP** et **transcriptomiques**.
Projet mêlant **science des données**, **biologie moléculaire** et **intelligence artificielle**.

## Structure du projet

```bash
BioResistanceAI/
├── config/                 # Fichiers de configuration YAML
├── data/                   # Données brutes et prétraitées (non versionnées)
├── pipelines/              # Orchestration
├── results/                # Résultats (non versionnés)
├── src/                    # Code source Python
│   ├── data/               # Prétraitement
│   ├── evaluation/         # Résultats
│   ├── features/           # Feature engineering, sélection de variables
    ├── models/             # Entrainement
│   └── utils/              # Fonctions utilitaires
├── tests/                  
└── requirements.txt        # Dépendances Python
```

## 🛠️ Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/NCSdecoopman/BioResistanceAI.git
cd BioResistanceAI
```

### 2. Créer un environnement virtuel

```bash
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate sous Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

## Configuration

Les modèles sont paramétrés via un fichier YAML dans `config/`.

Exemple :

```yaml
models:
  RandomForest:
    n_estimators: [100, 200]
    max_depth: [5, 10]
  LogisticRegression:
    penalty: ["l1", "l2"]
    solver: ["liblinear"]
```

## Utilisation

### Entraînement + évaluation (tous antibiotiques)

```bash
python -m pipelines.main_pipeline
```

### Résultats produits

* `partial_results.json` : scores, contributions, temps

## Exemple de sortie

| Antibiotique | Modèle         | Recall | GPA (%) | SNPs (%) | GenExp (%) |
| ------------ | -------------- | ------ | ------- | -------- | ---------- |
| AMX          | RandomForest   | 0.87   | 45.3    | 38.1     | 16.6       |
| CTX          | Logistic (FDR) | 0.92   | 71.2    | 12.8     | 16.0       |

## Références

* Documentation disponible : [https://ncsdecoopman.github.io/](https://ncsdecoopman.github.io/)
