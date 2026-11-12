"""Exploratory data analysis -> reports/figures/eda_*.png.   Run: python -m src.eda"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from .config import FIG_DIR, RAW_DIR, TARGET, HI_THRESHOLD_C, NUM_FEATURES
from .data_loader import load_dataset

C = {"orange": "#F97316", "red": "#E5484D", "blue": "#38BDF8", "navy": "#101B2D"}


def save(n):
    plt.tight_layout(); plt.savefig(FIG_DIR / n, dpi=150); plt.close()


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True); sns.set_theme(style="whitegrid")
    df = load_dataset(); df["month"] = df["date"].dt.month
    raw = pd.read_csv(RAW_DIR / "open_meteo.csv")

    plt.figure(figsize=(7, 4)); sns.histplot(df["temperature_2m_max"], bins=40, color=C["orange"], kde=True)
    plt.title("Distribution of daily maximum temperature (ERA5)"); plt.xlabel("deg C"); save("eda_01_temperature_distribution.png")

    plt.figure(figsize=(11, 9)); cols = [c for c in NUM_FEATURES if c not in ("month_sin", "month_cos", "doy_sin", "doy_cos")] + ["station_heat_index_c", TARGET]
    sns.heatmap(df[cols].corr(), cmap="coolwarm", center=0, annot=False, cbar_kws={"shrink": .7})
    plt.title("Correlation heatmap"); save("eda_02_correlation_heatmap.png")

    plt.figure(figsize=(12, 4)); plt.plot(df["date"], df["temperature_2m_max"], lw=.4, color=C["orange"])
    plt.plot(df["date"], df["temperature_2m_max"].rolling(365, min_periods=100).mean(), color=C["navy"], label="365-day mean")
    plt.title("Daily max temperature over time"); plt.legend(); save("eda_03_temperature_over_time.png")

    plt.figure(figsize=(4.5, 4)); vc = df[TARGET].value_counts().sort_index()
    plt.bar(["Safe (0)", "Dangerous (1)"], vc.values, color=[C["blue"], C["red"]])
    plt.ylim(0, max(vc.values) * 1.25)
    for i, v in enumerate(vc.values): plt.text(i, v + max(vc.values) * .015, f"{v:,}\n({v/len(df):.1%})", ha="center", va="bottom")
    plt.title("Class distribution", pad=18); save("eda_04_class_distribution.png")

    plt.figure(figsize=(11, 4)); sns.boxplot(data=df[["temperature_2m_max", "temperature_2m_min", "dew_point_2m_mean", "heat_index_era", "wind_speed_10m_max"]], palette="Oranges")
    plt.title("Outlier boxplots (extremes are kept: they ARE the heatwaves)"); save("eda_05_outlier_boxplots.png")

    plt.figure(figsize=(6.5, 5)); s = df.sample(min(len(df), 4000), random_state=1)
    sns.scatterplot(data=s, x="temperature_2m_max", y="relative_humidity_2m_mean", hue=TARGET, palette=[C["blue"], C["red"]], s=14, alpha=.6)
    plt.title("Temperature vs humidity coloured by label"); save("eda_06_temp_vs_humidity.png")

    plt.figure(figsize=(10, 4)); sns.heatmap(raw.isnull().T, cbar=False, yticklabels=True, xticklabels=False)
    plt.title("Missing values in raw Open-Meteo data (yellow/dark = missing)"); save("eda_07_missing_values.png")

    plt.figure(figsize=(9, 4)); sns.boxplot(data=df, x="month", y="heat_index_era", color=C["orange"])
    plt.axhline(HI_THRESHOLD_C, color=C["red"], ls="--"); plt.title("Heat index by month (ERA5-derived)"); save("eda_08_heat_index_by_month.png")

    plt.figure(figsize=(8, 4)); df.groupby("month")[TARGET].mean().mul(100).plot.bar(color=C["red"])
    plt.ylabel("% of days labelled dangerous"); plt.title("Dangerous-heat frequency by month"); save("eda_09_danger_by_month.png")

    plt.figure(figsize=(6, 5)); sns.scatterplot(data=s, x="heat_index_era", y="station_heat_index_c", s=10, alpha=.4, color=C["navy"])
    plt.title("ERA5 grid heat index vs airport station heat index"); save("eda_10_era5_vs_station.png")
    print(f"Saved 10 EDA figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
