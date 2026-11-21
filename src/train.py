"""Full training run: chronological split -> tune classifiers plus a baseline -> pick on VALIDATION
-> threshold on validation -> final one-shot test evaluation -> save pipeline + metadata + figures.
Run:  python -m src.train"""
import json, platform, sys
from datetime import datetime, timezone
import joblib
import pandas as pd
import sklearn
from . import evaluate as ev
from .config import (ALL_FEATURES, TARGET, MODEL_DIR, REPORT_DIR, RAW_DIR, TRAIN_FRAC, VAL_FRAC,
                     GAP_DAYS, HI_THRESHOLD_C, RANDOM_STATE)
from .data_loader import load_dataset
from .tuning import MODELS, tune


def chronological_split(df):
    """70/15/15 by time, with a 7-day purge before each boundary (labels look 7 days ahead)."""
    df = df.sort_values("date").reset_index(drop=True); n = len(df)
    d1, d2 = df["date"].iloc[int(TRAIN_FRAC * n)], df["date"].iloc[int((TRAIN_FRAC + VAL_FRAC) * n)]
    gap = pd.Timedelta(days=GAP_DAYS)
    return (df[df["date"] < d1 - gap], df[(df["date"] >= d1) & (df["date"] < d2 - gap)], df[df["date"] >= d2])


def main():
    MODEL_DIR.mkdir(exist_ok=True); REPORT_DIR.mkdir(exist_ok=True)
    df = load_dataset(); tr, va, te = chronological_split(df)
    split_info = {k: {"rows": len(d), "from": str(d["date"].min().date()), "to": str(d["date"].max().date()),
                      "positive_rate": round(float(d[TARGET].mean()), 4)} for k, d in [("train", tr), ("validation", va), ("test", te)]}
    print(json.dumps(split_info, indent=2))
    Xtr, ytr, Xva, yva, Xte, yte = tr[ALL_FEATURES], tr[TARGET], va[ALL_FEATURES], va[TARGET], te[ALL_FEATURES], te[TARGET]

    fitted, table = {}, {}
    for name in MODELS:
        print(f"\n=== {name} ===")
        pipe, info = tune(name, Xtr, ytr); fitted[name] = pipe
        ptr, pva = pipe.predict_proba(Xtr)[:, 1], pipe.predict_proba(Xva)[:, 1]
        m = ev.metrics_at(yva, pva)
        table[name] = {"best_params": (info or {}).get("best_params"), "cv_pr_auc": (info or {}).get("cv_pr_auc"),
                       "train_pr_auc": ev.metrics_at(ytr, ptr)["pr_auc"], "val_pr_auc": m["pr_auc"],
                       "val_f1": m["f1"], "val_recall": m["recall"], "val_roc_auc": m["roc_auc"]}
        print({k: v for k, v in table[name].items() if k != "best_params"})

    # ---- model selection uses VALIDATION only (never the test set) ----
    best = max((n for n in MODELS if "(baseline)" not in n), key=lambda n: table[n]["val_pr_auc"])   # baselines never selected
    pipe = fitted[best]
    thr = ev.best_threshold_f2(yva, pipe.predict_proba(Xva)[:, 1])
    print(f"\nSelected: {best} | threshold (max F2 on validation) = {thr:.3f}")

    # ---- final, one-shot TEST evaluation ----
    probas = {n: fitted[n].predict_proba(Xte)[:, 1] for n in MODELS}
    for n in MODELS:
        t = ev.metrics_at(yte, probas[n]); table[n].update({"test_pr_auc": t["pr_auc"], "test_f1": t["f1"],
                                                            "test_recall": t["recall"], "test_roc_auc": t["roc_auc"],
                                                            "test_precision": t["precision"], "test_accuracy": t["accuracy"]})
    test_default = ev.metrics_at(yte, probas[best], 0.5)
    test_tuned = ev.metrics_at(yte, probas[best], thr)
    pd.DataFrame(table).T.drop(columns=["best_params"]).round(4).to_csv(REPORT_DIR / "model_comparison.csv")

    # ---- figures ----
    ev.plot_pr_roc(yte, probas); ev.plot_confusion(yte, probas[best], thr, best)
    ev.plot_calibration(yte, probas[best], best); ev.plot_overfit(table)
    importance = ev.plot_feature_importance(pipe, Xva, yva)

    # ---- save artefacts ----
    collected = None
    meta_file = RAW_DIR / "collection_metadata.json"
    if meta_file.exists(): collected = json.loads(meta_file.read_text())["downloaded_utc"]
    metadata = {"project": "HeatWave AI", "task": "Binary classification", "location": "Karachi, Pakistan",
                "target": TARGET, "target_definition": f"1 if station heat index >= {HI_THRESHOLD_C} C on any of days t+5..t+7",
                "prediction_contract": "features of issue day t (and its past) -> label for t+5..t+7",
                "model": best, "best_params": table[best]["best_params"], "decision_threshold": thr,
                "primary_metric": "PR-AUC", "features": ALL_FEATURES, "feature_version": "1.0",
                "trained_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"), "data_downloaded_utc": collected,
                "dataset_rows": len(df), "split": split_info,
                "test_metrics_threshold_0.5": test_default, "test_metrics_tuned_threshold": test_tuned,
                "top_features": importance,
                "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "pandas": pd.__version__}}
    joblib.dump(pipe, MODEL_DIR / "heatwave_pipeline.joblib")
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    (REPORT_DIR / "metrics.json").write_text(json.dumps({"comparison": table, "selected": best, **{k: metadata[k] for k in
                                             ["decision_threshold", "split", "test_metrics_threshold_0.5", "test_metrics_tuned_threshold", "top_features"]}}, indent=2))
    print("\nSaved pipeline, metadata, metrics and figures.")
    print(pd.DataFrame(table).T[["cv_pr_auc", "train_pr_auc", "val_pr_auc", "test_pr_auc", "test_f1", "test_recall", "test_roc_auc"]].round(3))


if __name__ == "__main__":
    main()
