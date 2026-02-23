# Car Auction Machine Learning Project — Full Report

## 1. Project Overview

This project analyzes **used car auction data** (Kaggle: `tunguz/used-car-auction-prices`) to clean data, explore patterns, build regression and classification models, perform clustering, and detect anomalies. It is structured for portfolio use and demonstrates data science and ML workflows from raw data to insights and models.

---

## 2. Data

### 2.1 Source
- **Dataset**: Used car auction prices (`car_prices.csv`)
- **Origin**: Loaded via **kagglehub** from Kaggle dataset `tunguz/used-car-auction-prices`
- **Initial size**: ~558,811 rows × 16 columns (after cleaning: ~545,433 rows; see below)

### 2.2 Raw Data Fields
- **Identifiers**: `vin`, `seller`
- **Vehicle**: `year`, `make`, `model`, `trim`, `body`, `transmission`, `color`, `interior`
- **Market**: `state`, `condition`, `odometer`, `mmr` (Manheim Market Report), `sellingprice`, `saledate`

### 2.3 Data Locations
- **Raw**: `data/raw/car_prices.csv`
- **Cleaned**: `data/cleaned/car_prices_cleaned.csv`
- **External**: `data/raw/google_trends/` (e.g. Ford pickup, Toyota Camry, Nissan Altima) for trend analysis in some notebooks

---

## 3. Project Structure

```
Car-Auction-Machine-Learning-Project/
├── data/
│   ├── raw/                    # car_prices.csv, google_trends/*.csv
│   └── cleaned/                # car_prices_cleaned.csv
├── docs/                        # Documentation (e.g. this report)
├── notebooks/                   # Jupyter notebooks
├── src/                         # Reusable Python modules
│   ├── cleaning_scripts.py     # Body-type standardization
│   ├── kmeans.py               # Custom K-Means + plotting
│   ├── anomaly_detection.py    # Isolation Forest & LOF
│   └── archive/main.py         # Archived code
├── requirements.txt
└── README.md
```

---

## 4. Data Cleaning (Notebook 01 — `01_data_cleaning.ipynb`)

**Purpose**: Turn raw auction data into a single, ML-ready cleaned CSV.

### Steps
1. **Load**: Fetch dataset via kagglehub and save to `data/raw/car_prices.csv`.
2. **Inspect**: `.describe()`, `.info()`, null shares; note oddities (e.g. odometer 999,999, min price/odometer 1.0).
3. **Handle nulls**:
   - Fill: `transmission` → `'automatic'`, `interior`/`color` → `'black'`, `trim` → `'Base'`, `condition` → column mean.
   - Drop rows with nulls in `make`, `model`, `odometer`, `body`.
4. **Drop columns**: `vin`, `seller`, `trim` (identifier/redundant).
5. **Categorical encoding**:
   - `transmission`: normalize (e.g. `'sedan'`/`'Sedan'` → `'automatic'`), then one-hot encode.
   - `color`: replace `'—'` and numeric-only values with `'black'`.
   - `state`: keep as-is (later one-hot encoded).
   - `body`: standardize via **`cleaning_scripts.standardize_body`** (e.g. map many body strings to: suv, sedan, convertible, coupe, wagon, hatchback, truck, van).
6. **Dates**: `saledate` → datetime; derive `sale_year`, `sale_month`; drop `saledate`.
7. **One-hot encode**: `make`, `model`, `body`, `color`, `interior`, `state` → many binary columns (~920 total).
8. **Numeric**: Ensure `mmr` is numeric (`pd.to_numeric(..., errors='coerce')`).
9. **Export**: Save to `data/cleaned/car_prices_cleaned.csv`.

**Result**: One cleaned CSV with no object columns (all numeric/bool), no nulls, standardized body types, and expanded categoricals for modeling.

---

## 5. Exploratory and Graphical Analysis

### 5.1 Notebook 02 — `02_analysis.ipynb`
- **Role**: High-level **analysis playbook** and starter EDA.
- **Content**: Section outline for:
  - Price distribution, feature impact, geographic and time-based analysis
  - Market segments, depreciation, statistical testing, market efficiency (e.g. price vs MMR)
- **Code**: Loads cleaned data, sets style (e.g. Seaborn darkgrid), and includes examples such as:
  - Average selling price over time (monthly)
  - Average selling price by state (bar plot)

### 5.2 `first_exploration.ipynb`
- **Role**: Early **loading and cleaning** of the CSV (overlaps with 01; can be considered exploratory/legacy).

### 5.3 `analysis_of_observations.ipynb`
- **Role**: **Observation-driven analysis** of pricing and market dynamics.
- **Content**: Uses cleaned data and custom K-Means; addresses questions like “Why do cars with similar attributes sell for different prices?” with hypotheses (region, unrecorded features, negotiation/auction effects).

### 5.4 `graphical_analysis.ipynb`
- **Role**: **Visual analysis** of sales and **Google Trends**.
- **Content**: Combines cleaned car data with Google Trends CSVs (Ford pickup, Toyota Camry, Nissan Altima) to explore:
  - Selling price and condition trends
  - Popular models and correlation between search interest and sales

---

## 6. Regression and Classification (Notebook 04 — `car_price_prediction_and_condition_analysis.ipynb`)

**Purpose**: Predict **selling price** (regression) and **condition** (classification) with clear sections and markdown.

### 6.1 Regression: Price Prediction
- **Features**: `year`, `odometer`, `condition`, `mmr`
- **Target**: `sellingprice`
- **Splits**: 80/20 train/test, then 80/20 of train for train/validation (64% / 16% / 20%).
- **Models**:
  - **Basic**: Linear Regression, Random Forest Regressor, Decision Tree Regressor
  - **Regularized**: Ridge, Lasso, Elastic Net
- **Metrics**: RMSE, R² on validation and test.
- **Findings**: Random Forest best (e.g. test RMSE ~1507, R² ~0.98); all models show strong R² and little overfitting; adding `year` and `mmr` improved over odometer/condition only.

### 6.2 Classification: Condition Prediction
- **Features**: `year`, `odometer`, `mmr` (scaled with `StandardScaler`).
- **Target**: `condition` binned into Poor / Fair / Good (e.g. 0–2, 2–3, 3–5).
- **Models**:
  - **KNN**: k=5 and k=11 (multi-class); k=7 with `weights='distance'` for binary Good vs Not Good.
  - **Logistic Regression**: multi-class condition.
- **Metrics**: Accuracy, precision, classification report (per class).
- **Findings**: Logistic Regression best test accuracy (~0.69); KNN k=11 less overfit than k=5; all models strong on “Good,” weaker on “Poor”/“Fair”; binary task improves accuracy (~0.72) but with overfitting on train.

---

## 7. Gradient Boosting Regressor (Notebook 03 — `03_GradientBoostingRegressor.ipynb`)

**Purpose**: Predict **selling price** with **Gradient Boosting** on the **full cleaned feature set**.

### Setup
- **Features**: All columns except `sellingprice` (full one-hot encoded cleaned data).
- **Target**: `sellingprice`
- **Split**: 80/20 train/test, shuffled, `random_state=42`.

### Model
- **Algorithm**: `sklearn.ensemble.GradientBoostingRegressor`
- **Parameters**: e.g. `n_estimators=100`, `learning_rate=0.1`, `max_depth=20`, `min_samples_split=5`, `min_samples_leaf=3`, `max_features=0.6`, `loss='huber'`, `random_state=42`
- **Metric**: MAE (train and test).

### Result
- Train MAE ~418, test MAE ~898 (interpreted as ~\$898 average error vs average price ~\$13,725).

Notebook also explains hyperparameters (n_estimators, learning_rate, max_depth, etc.) and the boosting idea (residuals, weak learners, learning rate).

---

## 8. Clustering (`clustering.ipynb`)

**Purpose**: **Segment vehicles** using a **custom K-Means** implementation and optional visuals.

### Implementation (`src/kmeans.py`)
- **CustomKMeans**: Custom K-Means with configurable `k`, `max_iters`, `tol`, `random_state`; uses Euclidean distance, centroid updates, convergence check, and computes inertia (SSE).
- **plot_clusters**: Plot clusters and centroids (e.g. 2D subset of features or PCA).

### Usage
- **Data**: Cleaned car data (numeric columns only for K-Means); optionally Google Trends paths for the same datasets as in graphical analysis.
- **Goal**: Group cars by numeric features to reveal market segments (e.g. by price, mileage, condition).

---

## 9. Anomaly Detection (`anomaly_detection.ipynb` + `src/anomaly_detection.py`)

**Purpose**: Flag **outliers** in the auction data (e.g. data errors or unusual listings).

### Implementation (`src/anomaly_detection.py`)
- **Isolation Forest**: `detect_anomalies_iforest(df, features, contamination=0.001, n_estimators=100)` — scales features, fits Isolation Forest, returns anomaly/normal splits and adds `iforest_anomaly_label`.
- **Local Outlier Factor (LOF)**: `detect_anomalies_lof(df, features, contamination=0.001, n_neighbors=20)` — same idea with LOF, adds `lof_anomaly_label`.

### Notebook
- Loads data, selects features (e.g. condition, odometer, selling price), runs both methods, and inspects/visualizes anomalies.

---

## 10. Source Code Summary

| File | Role |
|------|------|
| `cleaning_scripts.py` | `standardize_body(body)` — maps body type strings to a small set (convertible, van, truck, coupe, sedan, wagon, hatchback, suv, other). |
| `kmeans.py` | Custom K-Means class (fit, centroids, inertia) and `plot_clusters` for visualization. |
| `anomaly_detection.py` | `detect_anomalies_iforest` and `detect_anomalies_lof`; scale features, fit model, return anomalies and normal data. |

---

## 11. Dependencies (`requirements.txt`)

- **Data**: `pandas>=1.0.0`, `numpy>=1.18.0`
- **Visualization**: `matplotlib>=3.0.0`, `seaborn>=0.11.0`

Not listed but used in notebooks: `scikit-learn`, `kagglehub` (for 01). Consider adding them to `requirements.txt` for reproducibility.

---

## 12. Notebooks at a Glance

| Notebook | Main purpose |
|----------|------------------|
| **01_data_cleaning** | Load from Kaggle, clean, encode, save cleaned CSV. |
| **02_analysis** | Analysis outline + starter EDA (e.g. price over time, by state). |
| **03_GradientBoostingRegressor** | Price prediction with full-feature Gradient Boosting; MAE. |
| **04** (car_price_prediction_and_condition_analysis) | Price regression (linear, tree, regularized) + condition classification (KNN, logistic); structured sections and summaries. |
| **clustering** | K-Means clustering on cleaned data using custom implementation. |
| **anomaly_detection** | Isolation Forest and LOF on selected features. |
| **first_exploration** | Early load/clean exploration. |
| **analysis_of_observations** | Observation-based pricing/market analysis + K-Means. |
| **graphical_analysis** | Plots + Google Trends (Ford, Camry, Altima). |

---

## 13. Insights and Takeaways (from README and notebooks)

- **Price drivers**: Mileage, condition, region (and MMR) are strong predictors; adding `year` and `mmr` improves regression.
- **Anomalies**: Often data entry errors or rare listings; flagged with Isolation Forest and LOF.
- **Clustering**: Reveals distinct segments (e.g. by price/condition/mileage).
- **Classification**: Condition (Poor/Fair/Good) is predictable but imbalanced; “Good” is easiest; binary Good vs Not Good improves accuracy at the cost of overfitting with current features.

---

## 14. How to Run (from README)

1. Clone repo, then: `pip install -r requirements.txt`
2. (Optional) Run cleaning: `python src/clean_data.py` — note that the main cleaning pipeline lives in **01_data_cleaning.ipynb** (and uses kagglehub); confirm whether `clean_data.py` exists or if “run data cleaning” means running notebook 01.
3. Run notebooks: `jupyter notebook notebooks/`

---

*Report generated from the current state of the Car-Auction-Machine-Learning-Project repository.*
