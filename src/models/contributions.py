# src/models/contributions.py

from sklearn.metrics import recall_score
from src.utils.predict import safe_predict

def compute_group_contributions(model, X_test, y_test, scaler, feature_types, recall_ref, random_state):
    """
    Calcule la contribution de chaque groupe de variables (gpa_, snps_, genexp_)
    en permutant aléatoirement les colonnes du groupe et en mesurant la chute de performance.
    Le scaler est appliqué uniquement à genexp_.

    Arguments :
    - model : classifieur entraîné
    - X_test : DataFrame d'entrée
    - y_test : cibles vraies
    - scaler : StandardScaler entraîné sur les colonnes genexp_
    - feature_types : dict {"gpa_": [...], "snps_": [...], "genexp_": [...]}
    - recall_ref : rappel de référence (modèle non perturbé)
    - random_state : entier pour reproductibilité

    Retour :
    - contributions_pct : contributions normalisées en pourcentage
    """
    contributions = {}
    genexp_cols = feature_types.get("genexp_", [])

    for group, cols in feature_types.items():
        # Copie et permutation du groupe
        X_mod = X_test.copy()
        X_mod[cols] = X_mod[cols].sample(frac=1.0, random_state=random_state).values

        # Appliquer le scaler uniquement aux colonnes genexp_
        X_mod_scaled = X_mod.copy()
        if genexp_cols:
            X_mod_scaled[genexp_cols] = scaler.transform(X_mod[genexp_cols])


        # *** PATCH pour Skorch ***
        # Si c'est un Skorch, on passe un numpy array float32
        is_skorch = "skorch" in str(type(model)).lower() or hasattr(model, "module_")
        X_to_pred = X_mod_scaled.values.astype("float32") if is_skorch else X_mod_scaled


        # Prédiction et score
        # Pour Skorch : float32 array, sinon DataFrame
        y_pred_mod = safe_predict(model, X_to_pred, str(type(model)))
        permuted_score = recall_score(y_test, y_pred_mod, average="macro")
        delta = recall_ref - permuted_score

        # Affichage si la permutation améliore le score (cas rare mais possible)
        if delta < 0:
            print(f"[INFO] Score ↑ après permutation du groupe {group} : {permuted_score:.4f} > {recall_ref:.4f}")

        contributions[group] = delta  # on garde la valeur brute, même négative

    # Normalisation en pourcentage
    total = sum(abs(val) for val in contributions.values())
    contributions_pct = {
        group: round(100 * abs(val) / total, 2) if total > 0 else 0.0
        for group, val in contributions.items()
    }

    return contributions_pct

