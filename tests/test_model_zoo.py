from src.tuning import MODELS


def test_classifiers_and_dummy_baseline_are_configured():
    baseline = "Dummy (baseline)"
    assert baseline in MODELS
    assert len(MODELS) == 7
    assert set(MODELS) - {baseline} == {
        "Logistic Regression",
        "K-Nearest Neighbors",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "SVM (RBF)",
    }


def test_knn_is_scaled_and_hyperparameter_tuned():
    _, scale, grid = MODELS["K-Nearest Neighbors"]
    assert scale is True
    assert set(grid) == {"classifier__n_neighbors"}
    assert len(grid["classifier__n_neighbors"]) >= 2


def test_decision_tree_has_depth_and_leaf_size_search():
    _, scale, grid = MODELS["Decision Tree"]
    assert scale is False
    assert set(grid) == {"classifier__max_depth", "classifier__min_samples_leaf"}
    assert len(grid["classifier__max_depth"]) >= 2
    assert len(grid["classifier__min_samples_leaf"]) >= 2
