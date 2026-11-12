# Graph guide

These notes describe what each chart shows and how to read it. Scores summarize the historical test period; they do not guarantee future predictions.

## Exploratory data analysis

| Figure | Plain-language interpretation |
|---|---|
| `eda_01_temperature_distribution.png` | Shows which daily high temperatures happened most often. The far ends are rare hot or cool days, and extreme heat is kept because it matters to this project. |
| `eda_02_correlation_heatmap.png` | Shows which measurements tend to rise or fall together. Strong relationships are clues about the data, not proof that one measurement causes another. |
| `eda_03_temperature_over_time.png` | Shows daily highs and a smoother year-scale line. The smooth line makes long-term patterns easier to see than the daily ups and downs. |
| `eda_04_class_distribution.png` | Counts safe and dangerous labels. This reveals how common each outcome is and why accuracy alone can be misleading. |
| `eda_05_outlier_boxplots.png` | Highlights unusual weather readings. Unusually hot values are inspected but not automatically removed: they may be real heat events. |
| `eda_06_temp_vs_humidity.png` | Each point is a day, coloured by its label. It helps show how temperature and humidity occur together on dangerous and safe days. |
| `eda_07_missing_values.png` | Marks missing readings in the raw weather table. Blank areas mean the source did not provide a value; the cleaning pipeline handles missing features. |
| `eda_08_heat_index_by_month.png` | Compares the spread of estimated heat index across months. The dashed line marks the project's chosen 40.6 °C threshold. |
| `eda_09_danger_by_month.png` | Shows the percentage of days labelled dangerous in each month. Higher bars indicate more frequent dangerous labels in that month. |
| `eda_10_era5_vs_station.png` | Compares estimated heat index from the weather grid with the airport station value. Differences are expected because the sources measure different places/products. |

## Model evaluation

| Figure | Plain-language interpretation |
|---|---|
| `eval_pr_curves.png` | Higher curves mean the model finds more dangerous days while making fewer false alerts. The dotted line is the dangerous-day rate in the test period. |
| `eval_roc_curves.png` | Curves closer to the upper-left indicate better separation of dangerous and safe days. The diagonal is a no-skill reference. |
| `eval_confusion_matrix.png` | Diagonal boxes are correct predictions. Off-diagonal boxes count missed dangerous days and unnecessary danger alerts. |
| `eval_calibration.png` | Points closer to the dotted diagonal mean predicted probabilities agree better with observed rates in groups of test predictions. This does not guarantee future calibration. |
| `eval_feature_importance.png` | Larger bars mean shuffling that feature reduced validation PR-AUC more. It shows which inputs the model relies on, not what causes heat. |
| `eval_overfitting.png` | Compares training and validation PR-AUC. A large training-over-validation gap can warn of overfitting; similar scores are more reassuring. Weather changes over time can also cause a gap. |

## Reading the fit check

The training/validation chart is a diagnostic, not a target to make the bars look identical. Model selection uses validation data; the newest test period is kept for final evaluation. No model can be guaranteed free of underfitting or overfitting. Results also need to be interpreted alongside the changing dangerous-label rate across time.

In the current run, Logistic Regression was selected using validation PR-AUC. Its training PR-AUC is about 0.898 and validation PR-AUC about 0.798, a gap of about 0.100. This noticeable train/validation difference warrants caution; it should not be described as “no overfitting.” The Decision Tree is tuned for depth and minimum leaf size, and its train/validation gap is also shown rather than hidden.

The dangerous-label rate also changes from 24.2% in validation to 47.6% in the test period. Because the weather periods differ, do not interpret a high test score as proof that the model will perform the same way in the future.
