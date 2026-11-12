import json
import ui
import pandas as pd
import streamlit as st
from src.config import REPORT_DIR, FIG_DIR

ui.setup("Model Insights")
st.title("Model insights")
mf = REPORT_DIR / "metrics.json"
if not mf.exists():
    st.error("metrics.json not found. Run `python -m src.train`."); st.stop()
m = json.loads(mf.read_text()); t = m["test_metrics_tuned_threshold"]
st.markdown(f"**Selected model:** {m['selected']} (chosen on validation PR-AUC) · decision threshold {m['decision_threshold']:.3f} (max F2 on validation)")
selected = m["comparison"][m["selected"]]
fit_gap = selected["train_pr_auc"] - selected["val_pr_auc"]
if fit_gap >= 0.08:
    fit_note = "Training score is noticeably higher than validation; this can indicate overfitting, although the two time periods may also differ."
elif fit_gap < -0.05:
    fit_note = "Validation score is higher than training; this can happen with time-varying weather patterns and is not, by itself, evidence of underfitting."
else:
    fit_note = "Training and validation scores are reasonably close; this check shows no large train-to-validation gap."
st.info(f"**Fit check (train vs. validation):** PR-AUC gap {fit_gap:+.3f}. {fit_note} This is a diagnostic, not proof that future forecasts will be accurate.")
c = st.columns(5)
for col, (k, lab) in zip(c, [("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"), ("recall", "Recall"), ("precision", "Precision"), ("f1", "F1")]):
    with col: ui.metric_card(f"Test {lab}", f"{t[k]:.3f}")
st.subheader("Model comparison")
tbl = pd.DataFrame(m["comparison"]).T[["cv_pr_auc", "train_pr_auc", "val_pr_auc", "test_pr_auc", "test_roc_auc", "test_f1", "test_recall"]].astype(float).round(3)
st.dataframe(tbl)
st.caption("Six tuned classifiers are compared with the Dummy baseline. Model selection used validation only; held-out test results are reported separately.")
interpretations = {
    "eval_pr_curves.png": "Higher curves are better: they mean the model finds more dangerous days without creating as many false alarms. The dotted line is the test-set danger-day rate.",
    "eval_confusion_matrix.png": "The diagonal boxes are correct predictions. The off-diagonal boxes show missed dangerous days and unnecessary alerts.",
    "eval_feature_importance.png": "Larger bars mean shuffling that input hurt validation performance more; this is a clue, not proof of cause.",
    "eval_calibration.png": "Closer to the dotted diagonal means predicted probabilities more closely match observed rates; this is not a guarantee for future weather.",
    "eval_roc_curves.png": "Curves nearer the upper-left separate dangerous from safe days better; the diagonal is a no-skill reference.",
    "eval_overfitting.png": "For each model, similar train and validation bars suggest less overfitting; a large gap is a warning to investigate.",
}
for row in [("eval_pr_curves.png", "eval_confusion_matrix.png"), ("eval_feature_importance.png", "eval_calibration.png"), ("eval_roc_curves.png", "eval_overfitting.png")]:
    cols = st.columns(2)
    for col, f in zip(cols, row):
        if (FIG_DIR / f).exists():
            col.image(str(FIG_DIR / f))
            col.caption(interpretations[f])
