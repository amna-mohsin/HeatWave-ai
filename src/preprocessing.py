"""sklearn preprocessing. Everything is fitted INSIDE the Pipeline, so CV folds never leak."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from .config import NUM_FEATURES, CAT_FEATURES, SKEWED


def log1p_clip(x):
    """Named module-level function (lambdas cannot be pickled by joblib)."""
    return np.log1p(np.clip(x, 0, None))


def make_preprocessor(scale: bool) -> ColumnTransformer:
    plain = [c for c in NUM_FEATURES if c not in SKEWED]
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    skew_steps = [("log", FunctionTransformer(log1p_clip, feature_names_out="one-to-one")),
                  ("impute", SimpleImputer(strategy="median"))]
    if scale:
        skew_steps.append(("scale", StandardScaler()))
    cat_steps = [("impute", SimpleImputer(strategy="most_frequent")),
                 ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    return ColumnTransformer([("num", Pipeline(num_steps), plain),
                              ("skew", Pipeline(skew_steps), SKEWED),
                              ("cat", Pipeline(cat_steps), CAT_FEATURES)])


def make_pipeline(classifier, scale: bool) -> Pipeline:
    return Pipeline([("preprocess", make_preprocessor(scale)), ("classifier", classifier)])
