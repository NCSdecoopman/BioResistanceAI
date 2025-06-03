# src/utils/json.py

import numpy as np

def make_json_serializable(obj):
    """Convertit récursivement les objets non-JSON-sérialisables en string (notamment les classes, types, objets numpy, etc.)"""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, type):
        return f"{obj.__module__}.{obj.__name__}"
    elif callable(obj):
        # Pour les fonctions, classes, etc.
        return str(obj)
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj
