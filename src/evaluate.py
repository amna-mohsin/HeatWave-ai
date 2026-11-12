"""Metrics and plots."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from .config import FIG_DIR, RANDOM_STATE


def metrics_at(y, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    return {"pr_auc": float(average_precision_score(y, proba)), "roc_auc": float(roc_auc_score(y, proba)),
            "accuracy": float(accuracy_score(y, pred)), "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred)), "f1": float(f1_score(y, pred)),
            "brier": float(brier_score_loss(y, proba)), "threshold": float(thr),
            "confusion_matrix": confusion_matrix(y, pred).tolist()}


def best_threshold_f2(y, proba):
    """Threshold maximising F2 on VALIDATION data (recall weighted 2x precision: a missed heatwave is worse)."""
    p, r, t = precision_recall_curve(y, proba)
    f2 = 5 * p[:-1] * r[:-1] / np.clip(4 * p[:-1] + r[:-1], 1e-9, None)
    return float(t[int(np.argmax(f2))])


def _save(name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(FIG_DIR / name, dpi=150); plt.close()


def plot_pr_roc(y, probas: dict):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for n, p in probas.items():
        pr, rc, _ = precision_recall_curve(y, p); ax.plot(rc, pr, label=f"{n} (AP={average_precision_score(y, p):.3f})")
    ax.axhline(y.mean(), ls=":", c="grey"); ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curves (test set)")
    ax.legend(fontsize=8); _save("eval_pr_curves.png")
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for n, p in probas.items():
        fpr, tpr, _ = roc_curve(y, p); ax.plot(fpr, tpr, label=f"{n} (AUC={roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], ls=":", c="grey"); ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves (test set)")
    ax.legend(fontsize=8); _save("eval_roc_curves.png")


def plot_confusion(y, proba, thr, name):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y, (proba >= thr).astype(int)), display_labels=["Safe", "Dangerous"]).plot(ax=ax, cmap="Oranges", colorbar=False)
    ax.set_title(f"{name} - test (thr={thr:.2f})"); _save("eval_confusion_matrix.png")


def plot_calibration(y, proba, name):
    fp, mp = calibration_curve(y, proba, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.8, 4.5)); ax.plot(mp, fp, "o-", c="#F97316"); ax.plot([0, 1], [0, 1], ":", c="grey")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed fraction dangerous", title=f"Calibration - {name}")
    _save("eval_calibration.png")


def plot_feature_importance(pipe, X_val, y_val, top=15):
    r = permutation_importance(pipe, X_val, y_val, scoring="average_precision", n_repeats=3, random_state=RANDOM_STATE, n_jobs=1)
    idx = np.argsort(r.importances_mean)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh([X_val.columns[i] for i in idx], r.importances_mean[idx], xerr=r.importances_std[idx], color="#F97316")
    ax.set(xlabel="Drop in PR-AUC when shuffled", title="Permutation feature importance (validation)")
    _save("eval_feature_importance.png")
    return {X_val.columns[i]: float(r.importances_mean[i]) for i in np.argsort(r.importances_mean)[::-1][:top]}


def plot_overfit(table):
    names = list(table); w = 0.34; x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k, (lab, key) in enumerate([("Train", "train_pr_auc"), ("Validation", "val_pr_auc")]):
        ax.bar(x + (k - 0.5) * w, [table[n][key] for n in names], w, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("PR-AUC"); ax.set_ylim(0, 1); ax.legend()
    ax.set_title("Training vs validation PR-AUC (fit diagnostic)")
    _save("eval_overfitting.png")
