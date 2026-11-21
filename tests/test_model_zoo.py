import numpy as np
from src.baselines import PersistenceClassifier
from src.tuning import MODELS

BASELINES = {"Dummy (baseline)", "Persistence (baseline)"}


def test_classifiers_and_baselines_are_configured():
    assert BASELINES <= set(MODELS)
    assert len(MODELS) == 7
    assert set(MODELS) - BASELINES == {
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "SVM (RBF)",
    }


def test_decision_tree_has_depth_and_leaf_size_search():
    _, scale, grid = MODELS["Decision Tree"]
    assert scale is False
    assert set(grid) == {"classifier__max_depth", "classifier__min_samples_leaf"}
    assert len(grid["classifier__max_depth"]) >= 2
    assert len(grid["classifier__min_samples_leaf"]) >= 2


def test_persistence_uses_only_todays_heat_index():
    X = np.zeros((6, 3)); X[:, 1] = [30, 32, 34, 41, 43, 45]
    y = np.array([0, 0, 0, 1, 1, 1])
    m = PersistenceClassifier(col=1).fit(X, y)
    assert 34 < m.threshold_ <= 41
    assert m.predict(X).tolist() == y.tolist()
    X2 = X.copy(); X2[:, [0, 2]] = 99          # other columns are ignored
    assert np.allclose(m.predict_proba(X), m.predict_proba(X2))
    assert np.all(np.diff(m.predict_proba(X)[:, 1]) > 0)   # ranking follows heat index
