# src/features/feature_groups.py
# Classe les colonnes du DataFrame X en trois groupes (gpa, snps, genexp

def get_feature_groups(X):
    return {
        "gpa": [c for c in X.columns if c.startswith("gpa_")],
        "snps": [c for c in X.columns if c.startswith("snp_")],
        "genexp": [c for c in X.columns if c.startswith("genexp_")]
    }
