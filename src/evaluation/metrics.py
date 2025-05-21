# src/evaluation/metrics.py

from sklearn.metrics import recall_score, make_scorer

def macro_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average="macro")

SCORERS = {
    "recall_macro": make_scorer(macro_recall)
}
