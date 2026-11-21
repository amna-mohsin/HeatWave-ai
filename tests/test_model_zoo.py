import numpy as np
from src.baselines import PersistenceClassifier
from src.tuning import MODELS

BASELINES = {"Dummy (baseline)", "Persistence (baseline)"}


def test_classifiers_and_baselines_are_configured():
    assert BASELINES <= set(MODELS)
    assert len(MODELS) == 6
    assert set(MODELS) - BASELINES == {
        "Logistic Regression",
        "Random Forest",
        "Gradient Boosting",
        "SVM (RBF)",
    }


def test_random_forest_depth_is_limited():
    _, scale, grid = MODELS["Random Forest"]
    assert scale is False
    assert all(d is not None for d in grid["classifier__max_depth"])


def test_persistence_uses_only_todays_heat_index():
    X = np.zeros((6, 3)); X[:, 1] = [30, 32, 34, 41, 43, 45]
    y = np.array([0, 0, 0, 1, 1, 1])
    m = PersistenceClassifier(col=1).fit(X, y)
    assert 34 < m.threshold_ <= 41
    assert m.predict(X).tolist() == y.tolist()
    X2 = X.copy(); X2[:, [0, 2]] = 99          # other columns are ignored
    assert np.allclose(m.predict_proba(X), m.predict_proba(X2))
    assert np.all(np.diff(m.predict_proba(X)[:, 1]) > 0)   # ranking follows heat index
