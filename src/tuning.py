"""Model zoo + hyper-parameter grids + time-series GridSearchCV."""
import os
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.svm import SVC
from .config import RANDOM_STATE, GAP_DAYS
from .baselines import PersistenceClassifier
from .preprocessing import make_pipeline

MODELS = {   # name -> (estimator, needs_scaling, param_grid)
    "Dummy (baseline)": (DummyClassifier(strategy="prior"), False, {}),
    "Persistence (baseline)": (PersistenceClassifier(), False, {}),   # today's heat index only, no ML
    "Logistic Regression": (LogisticRegression(max_iter=3000, random_state=RANDOM_STATE), True,
                            {"classifier__C": [0.1, 1]}),
    "Decision Tree": (DecisionTreeClassifier(random_state=RANDOM_STATE), False,
                      {"classifier__max_depth": [4, 8],
                       "classifier__min_samples_leaf": [10, 30]}),
    "Random Forest": (RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE), False,
                      {"classifier__n_estimators": [150, 300],
                       "classifier__max_depth": [8]}),
    "Gradient Boosting": (GradientBoostingClassifier(random_state=RANDOM_STATE), False,
                          {"classifier__n_estimators": [100, 150]}),
    "SVM (RBF)": (CalibratedClassifierCV(estimator=SVC(kernel="rbf"), method="sigmoid", cv=3, ensemble=False), True,
                  {"classifier__estimator__C": [0.5, 2]}),
}


def tune(name, X_train, y_train, n_splits=5):
    est, scale, grid = MODELS[name]
    pipe = make_pipeline(est, scale)
    if os.environ.get("HEATWAVE_FAST"):      # smoke-test mode only: 1 value per hyper-parameter
        grid = {k: v[:1] for k, v in grid.items()}
    if not grid:                              # baseline: nothing to tune
        return pipe.fit(X_train, y_train), None
    cv = TimeSeriesSplit(n_splits=n_splits, gap=GAP_DAYS)   # expanding window, 7-row gap
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="average_precision", n_jobs=1, refit=True, verbose=1)
    gs.fit(X_train, y_train)
    return gs.best_estimator_, {"best_params": {k.replace("classifier__", "").replace("estimator__", ""): v for k, v in gs.best_params_.items()},
                                "cv_pr_auc": float(gs.best_score_)}
