# Car Auction Price Prediction

I took a messy 558k row real-world dataset (used car auction prices from Kaggle) and built a rigorous price prediction pipeline. The work goes from raw data through cleaning and exploratory analysis to baseline models, then to a tuned Gradient Boosting model. The focus is on honest evaluation and understanding what actually drives used car prices.

## What's in the repo

The notebooks are numbered so you can follow the pipeline in order. Everything lives at the root so you don't have to dig.

- **01_data_cleaning.ipynb** – Load from Kaggle, handle nulls, standardize body types and categoricals, one-hot encode, export a single cleaned CSV (~545k rows, 920 features).
- **02_eda.ipynb** – Exploratory data analysis: price distributions, odometer/year/condition vs price, correlations, geography, and time trends. Sets up the intuition for the modeling step.
- **03_baseline_models.ipynb** – Linear, Ridge, Lasso, and Random Forest on the full feature set. Establishes a performance floor and saves results so the next notebook can use them.
- **04_gradient_boosting.ipynb** – The main model. Gradient Boosting with full evaluation: baseline comparison, feature importance, and error analysis (where the model gets it wrong by price band and mileage). Includes a short conclusion on strengths and limitations.
- **05_anomaly_detection.ipynb** – Isolation Forest and LOF on a subset of features to flag outliers (data errors or unusual listings). Kept as a supporting piece rather than the core story.
- **06_condition_classification.ipynb** – Bonus: predicting condition (Poor/Fair/Good) with KNN and Logistic Regression. Kept separate so the main narrative stays on price prediction.

Older notebooks and a previous project report are in `archive/` if you want to see the evolution of the project.

## Project structure

```
Car-Auction-Machine-Learning-Project/
├── 01_data_cleaning.ipynb
├── 02_eda.ipynb
├── 03_baseline_models.ipynb
├── 04_gradient_boosting.ipynb
├── 05_anomaly_detection.ipynb
├── 06_condition_classification.ipynb
├── data/
│   ├── raw/           # car_prices.csv (and optional Google Trends CSVs)
│   └── cleaned/       # car_prices_cleaned.csv, baseline_results.json
├── src/               # cleaning_scripts.py, kmeans.py, anomaly_detection.py
├── archive/           # Legacy notebooks and old report
├── requirements.txt
└── README.md
```

## How to run it

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the notebooks in order. Start with **01_data_cleaning.ipynb** (needs Kaggle access for the dataset). It writes `data/raw/car_prices.csv` and `data/cleaned/car_prices_cleaned.csv`. The rest of the notebooks read from those.
3. Notebook 03 writes `data/baseline_results.json`; notebook 04 loads it so the baseline comparison table stays in sync without manual copy-paste.

If you already have the cleaned CSV from a previous run, you can start from 02 or 03.

## Results

Numbers below are from a single full run (train/test 80/20, `random_state=42`).

**Gradient Boosting (held-out test set):** MAE $898.34, R² 0.9731. Test MAE is about 6.5% of average selling price. RMSE is computed in notebook 04.

**Model comparison (test set):**

| Model              | Test MAE ($) | Test R²  |
| ------------------ | ------------ | -------- |
| Gradient Boosting  | 898.34       | 0.9731   |
| Random Forest      | 954.36       | 0.9712   |
| Lasso Regression   | 968.98       | 0.9704   |
| Ridge Regression   | 969.70       | 0.9705   |
| Linear Regression  | 969.87       | 0.9704   |

**Top features:** The top 5 features account for 98.1% of total importance. Run the feature importance section in notebook 04 to see the exact list (mmr, odometer, year, and condition are the main drivers).

**Where the model fails most:** Error is highest for high-end vehicles ($50k+), with test MAE about $3,808 on that segment (only 662 samples). By mileage, MAE is highest in the 0–25k mile segment (about $1,042) and lowest for 150k+ miles (about $662). The model does best in mid-range prices ($5k–$20k) and in high-mileage segments where there is more data.

## Takeaways

Price is driven mainly by MMR, odometer, year, and condition; the full encoded feature set lets Gradient Boosting beat the baselines. For more detail and numbers, see the Results section above. Anomaly detection (notebook 05) surfaces likely data errors and unusual listings.
