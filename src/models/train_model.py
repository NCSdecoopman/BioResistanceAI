import numpy as np
from skorch.classifier import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV
from src.evaluation.metrics import SCORERS

def train_with_gridsearch(model_class, params, X_train, y_train, cv, n_jobs, scoring):
    scorer = SCORERS.get(scoring, scoring)
    estimator = model_class(**{k: v[0] if isinstance(v, list) else v 
                               for k, v in params.items()})
    clf = GridSearchCV(
        estimator, params,
        scoring=scorer,
        cv=cv,
        n_jobs=n_jobs,
        error_score="raise",
    )

    # Si c'est un Skorch NeuralNetClassifier, on cast X en float32 et y en int64
    if isinstance(estimator, NeuralNetClassifier):
        X_fit = (X_train.values.astype(np.float32)
                 if hasattr(X_train, "values") else
                 X_train.astype(np.float32))
        y_fit = (y_train.values.astype(np.int64)
                 if hasattr(y_train, "values") else
                 y_train.astype(np.int64))
        clf.fit(X_fit, y_fit)
    else:
        clf.fit(X_train, y_train)

    return clf
