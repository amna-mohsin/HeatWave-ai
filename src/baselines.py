"""Persistence baseline: "hot today -> dangerous heat next week", using only today's ERA5 heat index."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score
from .config import NUM_FEATURES, SKEWED

# Position of heat_index_era in the preprocessor output (unscaled: plain numeric block comes first).
HEAT_INDEX_COL = [c for c in NUM_FEATURES if c not in SKEWED].index("heat_index_era")


class PersistenceClassifier(BaseEstimator, ClassifierMixin):
    """No machine learning: learns one heat-index cut-off on training data (max F1) and predicts
    "dangerous" whenever today's heat index is at or above it. predict_proba is a monotone
    transform of today's heat index, so ranking metrics (PR-AUC, ROC-AUC) score the raw heat index,
    and probability >= 0.5 exactly when the heat index >= the learned cut-off."""

    def __init__(self, col=HEAT_INDEX_COL):
        self.col = col

    def fit(self, X, y):
        h, y = np.asarray(X)[:, self.col].astype(float), np.asarray(y)
        candidates = np.unique(np.round(np.quantile(h, np.linspace(0.05, 0.95, 91)), 2))
        self.threshold_ = float(max(candidates, key=lambda t: f1_score(y, (h >= t).astype(int))))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        h = np.asarray(X)[:, self.col].astype(float)
        p = 1.0 / (1.0 + np.exp(-(h - self.threshold_)))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
