import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
OUTPUT = REPORTS / "HeatWave_AI_Report.pdf"
NAVY = "#14243D"
ORANGE = "#EA580C"
GREY = "#526176"


def paragraph(ax, text, x, y, width=92, size=9, line_height=0.024, color=NAVY):
    wrapped = textwrap.fill(text, width=width)
    ax.text(x, y, wrapped, transform=ax.transAxes, va="top", fontsize=size,
            color=color, linespacing=1.35)
    return y - line_height * (wrapped.count("\n") + 1)


def heading(ax, text, y):
    ax.text(0.07, y, text, transform=ax.transAxes, fontsize=14,
            fontweight="bold", color=NAVY, va="top")
    return y - 0.045


def image(ax, filename, bounds):
    path = FIGURES / filename
    if path.exists():
        from PIL import Image
        x, y, w, h = bounds
        ax.imshow(Image.open(path), extent=(x, x + w, y, y + h), aspect="auto")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)


def new_page(pdf, title, page):
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.07, 0.955, title, transform=ax.transAxes, fontsize=22,
            fontweight="bold", color=NAVY, va="top")
    ax.text(0.07, 0.025, f"HeatWave AI · CT-378 Machine Learning · Page {page}/4",
            transform=ax.transAxes, fontsize=8, color=GREY)
    return fig, ax


def build():
    metrics = json.loads((REPORTS / "metrics.json").read_text())
    quality = json.loads((REPORTS / "data_quality.json").read_text())
    selected = metrics["comparison"][metrics["selected"]]
    split = metrics["split"]

    with PdfPages(OUTPUT) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        ax.text(.07, .95, "HeatWave AI", transform=ax.transAxes, fontsize=28,
                fontweight="bold", color=NAVY, va="top")
        ax.text(.07, .91, "Week-Ahead Dangerous Heat Warning for Karachi",
                transform=ax.transAxes, fontsize=14, color=ORANGE, va="top")
        y = .85
        y = heading(ax, "Abstract", y)
        y = paragraph(ax, f"This supervised binary-classification project estimates whether Karachi's station heat index will reach 40.6 °C or higher on any of days 5–7 after an issue day. It combines Open-Meteo ERA5 weather features with NOAA GSOD station labels. The final dataset contains {quality['final_rows']:,} labelled daily records and {quality['n_features']} input features. {metrics['selected']} was selected using validation PR-AUC. This is an educational prototype, not an official warning service.", .07, y)
        y -= .018
        y = heading(ax, "Problem and data", y)
        y = paragraph(ax, "The aim is to explore whether historical weather patterns can provide useful advance signals of dangerous heat. Open-Meteo Historical Weather API supplies daily Karachi weather features (ERA5, CC BY 4.0); NOAA NCEI GSOD station 41780099999 supplies airport observations used to calculate the target heat index. Collection date, source links, citations, and data-processing notes are in data/DATA_SOURCES.md.", .07, y)
        y -= .015
        y = paragraph(ax, "Target: danger_5to7d = 1 when station heat index is at least 40.6 °C on any of days t+5, t+6, or t+7; otherwise 0 when all three observations are known. This is the project's selected threshold, not an official NOAA danger-category boundary.", .07, y)
        y -= .02
        y = heading(ax, "Data exploration", y)
        image(ax, "eda_04_class_distribution.png", (.18, .12, .64, .25))
        paragraph(ax, "The chart counts the two target classes. The target is not perfectly balanced, which is why the project reports precision-recall performance as well as other classification metrics.", .07, .095, width=100, size=8)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        fig, ax = new_page(pdf, "Method and model comparison", 2)
        y = .89
        y = paragraph(ax, "ERA5 and GSOD records are cleaned and joined by calendar date. Fahrenheit values and GSOD missing-value sentinels are handled; impossible values are checked while genuine heat extremes are retained. The model uses 29 engineered and source weather features. Preprocessing (imputation, precipitation transform, scaling where needed, and season encoding) is fitted inside each scikit-learn Pipeline.", .07, y)
        y -= .025
        y = heading(ax, "Time-aware validation", y)
        y = paragraph(ax, f"Data is split chronologically with a 7-day purge: training {split['train']['from']}–{split['train']['to']} ({split['train']['rows']:,} rows), validation {split['validation']['from']}–{split['validation']['to']} ({split['validation']['rows']:,}), and held-out test {split['test']['from']}–{split['test']['to']} ({split['test']['rows']:,}). Each tuned classifier uses five-fold expanding-window TimeSeriesSplit with a 7-row gap. PR-AUC is the tuning score; model selection and threshold choice use validation data.", .07, y)
        y -= .025
        y = heading(ax, "Results", y)
        names = list(metrics["comparison"])
        table = []
        for name in names:
            row = metrics["comparison"][name]
            table.append([name, "—" if row["cv_pr_auc"] is None else f"{row['cv_pr_auc']:.3f}",
                          f"{row['val_pr_auc']:.3f}", f"{row['test_pr_auc']:.3f}",
                          f"{row['test_recall']:.3f}"])
        tbl = ax.table(cellText=table,
                       colLabels=["Model", "CV PR-AUC", "Val PR-AUC", "Test PR-AUC", "Test recall"],
                       cellLoc="center", colLoc="center", bbox=[.07, .16, .86, .36],
                       colWidths=[.34, .15, .16, .17, .15])
        tbl.auto_set_font_size(False); tbl.set_fontsize(7)
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor(NAVY); cell.set_text_props(color="white", weight="bold")
            elif r % 2 == 0:
                cell.set_facecolor("#F1F5F9")
        result = metrics["test_metrics_tuned_threshold"]
        ax.text(.07, .115,
                f"Selected: {metrics['selected']} · threshold {metrics['decision_threshold']:.3f} (chosen on validation) · "
                f"test ROC-AUC {result['roc_auc']:.3f} · F1 {result['f1']:.3f} · "
                f"precision {result['precision']:.3f} · recall {result['recall']:.3f}",
                transform=ax.transAxes, fontsize=8, color=NAVY, va="top", wrap=True)
        ax.text(.07, .075, "Dummy is a no-learning baseline; the six other rows are tuned classifiers.",
                transform=ax.transAxes, fontsize=8, color=GREY, va="top")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = new_page(pdf, "Evaluation and fit diagnostics", 3)
        image(ax, "eval_pr_curves.png", (.06, .53, .42, .30))
        image(ax, "eval_confusion_matrix.png", (.53, .53, .40, .30))
        image(ax, "eval_overfitting.png", (.06, .17, .42, .30))
        image(ax, "eval_calibration.png", (.53, .17, .40, .30))
        ax.text(.07, .49, "PR curve: higher is better; dotted line is the test danger rate.",
                transform=ax.transAxes, fontsize=8, color=GREY, va="top")
        ax.text(.54, .49, "Confusion matrix: off-diagonal cells are missed events or false alarms.",
                transform=ax.transAxes, fontsize=8, color=GREY, va="top")
        ax.text(.07, .13, "Fit check: train-to-validation gaps can signal overfitting.",
                transform=ax.transAxes, fontsize=8, color=GREY, va="top")
        ax.text(.54, .13, "Calibration: closer to the dotted line means better probability agreement.",
                transform=ax.transAxes, fontsize=8, color=GREY, va="top")
        gap = selected["train_pr_auc"] - selected["val_pr_auc"]
        ax.text(.07, .075,
                f"{metrics['selected']} train PR-AUC {selected['train_pr_auc']:.3f}; validation PR-AUC "
                f"{selected['val_pr_auc']:.3f}; gap {gap:.3f}. This is a noticeable gap and warrants caution. "
                f"Danger-label prevalence also shifts from {split['validation']['positive_rate']:.1%} in validation "
                f"to {split['test']['positive_rate']:.1%} in test. Metrics may not carry over to future weather.",
                transform=ax.transAxes, fontsize=8, color=NAVY, va="top", wrap=True)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = new_page(pdf, "Deployment, limitations, and references", 4)
        y = .89
        y = heading(ax, "Deployment checklist", y)
        y = paragraph(ax, "Local applications: Streamlit dashboard and FastAPI + HTML endpoint. Cloud items are not published yet: add the real public Streamlit Community Cloud URL and the Hugging Face model/Space URLs after publishing and testing an actual prediction. Deployment screenshots also need to be captured from the live services.", .07, y)
        y -= .02
        y = heading(ax, "Limitations", y)
        y = paragraph(ax, "ERA5 is a gridded reanalysis while NOAA GSOD is one airport station; their measurements differ. The app's current-weather API product differs from the historical ERA5 training product. Dangerous-label prevalence changes substantially across validation and test time periods. Probabilities are model estimates, not guarantees, and HeatWave AI is not an official public warning system; follow Pakistan Meteorological Department advisories.", .07, y)
        y -= .02
        y = heading(ax, "Team contributions (complete with actual information)", y)
        y = paragraph(ax, "Hamna Ali Khan (CT-24157) — [actual tasks].\nAmna Mohsin (CT-24169) — [actual tasks].", .07, y, width=95)
        y -= .02
        y = heading(ax, "References", y)
        for ref in [
            "Open-Meteo Historical Weather API: https://open-meteo.com/en/docs/historical-weather-api",
            "NOAA NCEI GSOD: https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/",
            "NWS Heat Index equation: https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml",
            "Scikit-learn user guide: https://scikit-learn.org/stable/user_guide.html",
        ]:
            y = paragraph(ax, ref, .07, y, width=105, size=8, line_height=.021)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"Created {OUTPUT}")


if __name__ == "__main__":
    build()
