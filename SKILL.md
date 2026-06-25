# ds-pipeline skill

You are executing the `/ds-pipeline` skill.

## What this skill does

Given a dataset file and a modelling goal, you generate four production-ready files that mirror the proven pipeline:

```
<dataset>
    └── eda.ipynb          (data exploration & cleaning)
           └── features.ipynb     (feature engineering)
                  ├── modelling.ipynb    (ML model comparison + SHAP)
                  └── app.py             (Streamlit interactive dashboard)
```

---

## Step 0 — Collect inputs

If the user has not already provided all of the following, ask for them before doing anything else:

| Input | Example |
|---|---|
| **Dataset file path** | `bookings.csv`, `data/sales.db` |
| **Target variable name** | `NoShow`, `churn`, `price` |
| **Target description** | "predict whether a hotel guest will no-show" |
| **Task type** (optional — infer if not given) | `classification` or `regression` |

Infer task type from the target: binary/low-cardinality integer or Yes/No string → `classification`; continuous float or wide-range integer → `regression`.

---

## Step 1 — Explore the dataset

Run Python to profile the dataset before writing any notebook. Collect:

```python
import pandas as pd
df = pd.read_csv("<path>")           # or sqlite3 / pd.read_excel as appropriate
print(df.shape)
print(df.dtypes)
print(df.head(3))
print(df.isnull().sum())
for col in df.select_dtypes("object").columns:
    print(col, df[col].nunique(), df[col].value_counts().head(5).to_dict())
print(df.describe())
```

From this, identify:
- **Categorical columns** (dtype object / low-cardinality int)
- **Numerical columns** (int/float, high cardinality)
- **Date/time columns** (parse later for feature engineering)
- **ID columns** (drop — unique per row, no predictive value)
- **Columns with >40% missing** (flag for special handling or drop)
- **Target column** — confirm it matches what the user said

---

## Step 2 — Generate `eda.ipynb`

Write a Jupyter notebook with these sections. Adapt every cell to the **actual columns** found in Step 1 — never hardcode column names from the hotel example.

### Required sections

**1. Setup & Data Loading**
- Import pandas, numpy, matplotlib, seaborn
- Load dataset (CSV / SQLite / Excel as appropriate)
- Print shape, dtypes

**2. Initial Overview**
- `df.info()`, `df.describe()`
- Missing value heatmap or bar chart

**3. Target Variable Distribution**
- Classification: bar/pie chart of class counts, print class balance
- Regression: histogram + KDE, print mean/median/std

**4. Categorical Column Exploration**
- For each categorical column: value_counts bar chart, note any inconsistencies (mixed case, mixed types, unexpected values)

**5. Numerical Column Exploration**
- Histograms grid for all numerical columns
- Correlation heatmap (numerical columns only)
- Boxplots of key numericals vs target (classification) or scatter vs target (regression)

**6. Data Cleaning**
Generate cleaning cells specific to the issues found in Step 1:
- Mixed-type columns (numbers stored as strings, words vs digits) → standardise
- Inconsistent capitalisation → `.str.title()` or `.str.lower()`
- Negative values that should be positive → take absolute value or set NaN
- Currency/unit prefixes in numeric fields → strip and convert
- Date strings → `pd.to_datetime()`
- Columns with >40% missing → drop or impute depending on importance
- Drop duplicate rows if present
- Drop ID/key columns with unique values per row

**7. Save cleaned dataset**
```python
df.to_csv("cleaned_data.csv", index=False)
print(f"Saved {len(df):,} rows to cleaned_data.csv")
```

---

## Step 3 — Generate `features.ipynb`

**Input:** `cleaned_data.csv`
**Output:** `fea_table.csv`

### Required sections

**1. Load cleaned data**

**2. Feature Engineering**
Engineer features appropriate to the domain. Use judgment based on the columns present:

- **Date/time columns** → extract day-of-week, month, hour, time-since-reference (days)
- **Two date columns** → compute difference in days (e.g., lead time, duration)
- **Categorical columns with many levels** → target encode or frequency encode if >20 levels; otherwise leave for one-hot in modelling
- **Numerical pairs** → create ratio or product if it makes domain sense (e.g., price × duration = total value)
- **Text columns** → length, word count, or keyword flags (keep it simple)
- **Geo columns** → country/city grouping by continent or region if useful

For every engineered feature, print a `.describe()` or `.value_counts()` and a quick plot to sanity-check it.

**3. Final column selection**
```python
fea_table = df[[<target>, <selected feature columns>]]
print(fea_table.shape)
fea_table.head(3)
```
Exclude: raw date strings used only for engineering, ID columns, columns that are direct leakage of the target.

**4. EDA on fea_table**

- **Cohort analysis**: for each categorical feature, compute `fea_table.groupby(<col>)[<target>].mean()` and plot a bar chart (classification) or box plot (regression). Cross-tab the two most important categorical features as a heatmap.
- **Numerical feature distributions**: histogram grid coloured by target class (classification) or scatter matrix vs target (regression)

**5. Save**
```python
fea_table.to_csv("fea_table.csv", index=False)
```

---

## Step 4 — Generate `modelling.ipynb`

**Input:** `fea_table.csv`

### Required sections

**1. Load & split**
```python
from sklearn.model_selection import train_test_split
X = df.drop("<target>", axis=1)
y = df["<target>"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**2. Preprocessing pipeline**
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
```
- Categorical → SimpleImputer(strategy="most_frequent") + OneHotEncoder(handle_unknown="ignore")
- Numerical → SimpleImputer(strategy="median") + StandardScaler

**3. Models**

*Classification task:*
- Logistic Regression (baseline) — print coefficients
- XGBoost (`XGBClassifier`) — main model
- Random Forest (`RandomForestClassifier`) — ensemble comparison

*Regression task:*
- Linear Regression (baseline) — print coefficients
- XGBoost (`XGBRegressor`) — main model
- Random Forest (`RandomForestRegressor`) — ensemble comparison

For each model build a `Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])`, fit on train, predict on test.

**4. Evaluation**

*Classification:*
- Accuracy, Precision, Recall, F1, ROC AUC
- Confusion matrix heatmap
- ROC curves (all three models overlaid)

*Regression:*
- MAE, RMSE, R²
- Actual vs Predicted scatter plot
- Residual distribution histogram

Collect all metrics into a results DataFrame and display sorted by the primary metric (ROC AUC or R²).

**5. SHAP interpretability (XGBoost and Random Forest)**
```python
import shap
explainer = shap.TreeExplainer(pipeline.named_steps["model"])
X_test_processed = pipeline.named_steps["preprocessor"].transform(X_test)
shap_values = explainer.shap_values(X_test_processed)
shap.summary_plot(shap_values, X_test_processed, feature_names=feature_names)
```
Generate SHAP summary plot (bar) for both tree models.

**6. Model comparison summary**
Print/display the results table and state which model performed best and why.

---

## Step 5 — Generate `app.py`

Build a Streamlit dashboard using `plotly.express` for all charts. The dashboard must:

**Layout:**
- `st.set_page_config(layout="wide")`
- Sidebar with `st.multiselect` / `st.selectbox` / `st.radio` filters for the main categorical columns (typically 3–5 filters)
- All charts react to the sidebar filters via a single filtered DataFrame

**KPI row** (use `st.columns`):
- Total records count
- Target rate / mean (e.g., no-show rate, churn rate, avg price)
- 2–3 other headline stats meaningful to the domain

**Three tabs:**

*Tab 1 — Cohort Analysis*
- Bar chart of target rate / mean by each categorical dimension (one chart per categorical feature, 2-column grid)
- Heatmap of the two most important categorical features crossed
- Expandable full cohort data table with colour gradient

*Tab 2 — [Domain metric] Estimation*  
(Adapt name to the domain: CLV, Revenue, Score, etc.)
- If classification: expected value per record = some revenue/value column × P(positive class from segment)
- If regression: direct distribution of the predicted value
- Scatter of value vs risk/probability coloured by a categorical feature
- Segment summary table

*Tab 3 — Risk / Opportunity Scoring*
- Assign tiers (Low / Medium / High / Critical for risk, or Bronze/Silver/Gold for opportunity) using `pd.cut` on the target rate or predicted value
- Tier distribution bar chart
- Heatmap per main categorical split
- Drill-down ranked bar chart controlled by two independent `st.selectbox` widgets
- Full scored table with gradient colouring

**Data loading:** Always use `@st.cache_data`.  
**Chart sizing:** Always set explicit `height=` in `fig.update_layout()`.

---

## Step 6 — Final checks

After generating all four files:

1. Run a Python smoke test on the core data logic:
```python
import pandas as pd
df = pd.read_csv("fea_table.csv")
# Verify key computed columns exist and have sensible ranges
# Verify no NaN in critical columns
```

2. Start the Streamlit server and hit the health endpoint:
```bash
streamlit run app.py --server.headless true --server.port 8504 &
sleep 5 && curl -s http://localhost:8504/_stcore/health
```

3. Report to the user:
   - Files created (with line counts)
   - Any data quality issues found during Step 1
   - Key finding from the cohort analysis (highest/lowest risk segment, class imbalance if classification, target distribution if regression)
   - How to run: `jupyter notebook` for notebooks, `streamlit run app.py` for dashboard

---

## Rules

- **Never hardcode column names from the hotel example** (NoShow, Branch, country, etc.) — always derive from the actual dataset.
- **Never assume CSV** — check the file extension and use the right loader (`.db` → sqlite3, `.xlsx` → `pd.read_excel`, `.parquet` → `pd.read_parquet`).
- **Never skip Step 1** — the profile run is mandatory before writing any notebook cell.
- **Adapt section titles to the domain** — use the user's target description in headings, not generic placeholders.
- **Keep notebooks runnable top-to-bottom** — each notebook reads its input file by relative path (`"cleaned_data.csv"`, `"fea_table.csv"`) so they work in any directory.
- **Use `@st.cache_data`** on all data loading functions in `app.py`.
- **Use plotly for all dashboard charts** — not matplotlib/seaborn.
- If the dataset has >1M rows, add a `.sample(500_000, random_state=42)` note in the modelling notebook and explain it to the user.
