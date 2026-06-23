# Hotel No-Show Analytics — EDA & Interactive Dashboard

End-to-end analysis of hotel booking no-show patterns across 97,777 records, covering cohort analysis, simplified CLV estimation, risk scoring, and an interactive Streamlit dashboard.

---

## Project Structure

```
.
├── fea_table.csv          # Source data (97,777 bookings)
├── eda_notebook.ipynb     # EDA: cohort analysis, CLV, risk scoring
├── app.py                 # Streamlit interactive dashboard
└── README.md
```

---

## Dataset — `fea_table.csv`

| Column | Type | Description |
|---|---|---|
| `NoShow` | int | 1 = no-show, 0 = showed up |
| `Branch` | str | Hotel branch: Changi / Orchard |
| `country` | str | Guest country of origin (7 countries) |
| `first_time` | str | First booking? Yes / No |
| `RoomType` | str | Single / Queen / King / President Suite |
| `platform` | str | Booking channel: Website / Agent / Email / Phone |
| `Price_SGD` | float | Nightly rate (SGD) |
| `Length_of_Stay_Days` | int | Number of nights |
| `Lead_Time_Days` | int | Days between booking and check-in |
| `Estimated_Booking_Value` | float | Price × nights |
| `num_children` | float | Number of children in booking |
| `NumAdults` | int | Number of adults |

97,777 rows · 12 columns · no missing values.

---

## EDA Notebook — `eda_notebook.ipynb`

Run in Jupyter:

```bash
jupyter notebook eda_notebook.ipynb
```

### Contents

**Section 1 — Data Loading & Overview**
Shape, dtypes, descriptive statistics, missing value check, unique values per categorical column.

**Section 2 — Overall No-Show Distribution**
Pie chart showing 37.1% overall no-show rate (36,277 no-shows vs 61,500 arrivals).

**Section 3 — Cohort Analysis**
No-show rate broken down by every dimension, then crossed:

- Bar charts: Branch, Country, Platform, First-Time, Room Type
- Heatmaps: Branch × Country, Platform × First-Time, Branch × Room Type
- Four-way cohort table: Branch × Country × Platform × First-Time — top 10 highest and lowest risk cells

**Section 4 — Simplified CLV Estimation**

Formula used:
```
CLV = (Price_SGD × Length_of_Stay_Days × P_show) / churn_rate
```
- `P_show` = 1 − segment no-show rate
- `churn_rate` = segment first-time guest rate (loyalty proxy, clipped to [0.01, 0.99])

Outputs: segment-level CLV table, CLV bar chart by Branch & Country, individual CLV distribution histogram.

**Section 5 — Risk Scoring**

Each Branch × Country × Room Type cell scored by observed no-show rate and assigned a tier:

| Tier | No-Show Rate |
|---|---|
| Low | < 25% |
| Medium | 25 – 40% |
| High | 40 – 55% |
| Critical | > 55% |

Outputs: heatmaps per branch, tier distribution bar chart.

**Section 6 — Key Findings Summary**

| Finding | Detail |
|---|---|
| Overall no-show rate | 37.1% |
| Highest-risk cohort | Changi / China — 64.8% |
| Lowest-risk cohort | Orchard / Japan — 11.7% |
| First-time vs returning | 37.9% vs 14.4% (2.6× gap) |
| Platform effect | Negligible (36–38% across all channels) |
| Best CLV segment | Orchard / Japan |
| Worst CLV segment | Changi / China |

---

## Interactive Dashboard — `app.py`

### Installation

```bash
pip install pandas numpy streamlit plotly
```

### Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Requires `fea_table.csv` in the same directory.

### Sidebar — Drill-Down Filters

All charts update live when any filter changes:

| Filter | Options |
|---|---|
| Branch | Changi, Orchard |
| Country | 7 guest origin countries |
| Room Type | Single, Queen, King, President Suite |
| Booking Platform | Website, Agent, Email, Phone |
| Guest Type | All / First-Time Only / Returning Only |

### KPI Row

Five headline metrics computed on the filtered data: total bookings, no-show rate (with delta vs overall baseline), avg booking value, avg CLV, avg length of stay.

### Tab 1 — Cohort Analysis

- Bar charts: no-show rate by Branch, Country, Platform, First-Time vs Returning
- Heatmap: Branch × Country cross-tab
- Expandable table: full Branch × Country × Platform × First-Time cohort with colour-coded risk tiers

### Tab 2 — CLV Estimation

- Grouped bar chart: avg CLV by Branch & Country
- Bubble scatter: CLV vs no-show rate (bubble = booking volume)
- Scatter: Price × Length of Stay coloured by CLV, faceted by Branch
- Expandable segment CLV table with formatting

### Tab 3 — Risk Scoring

- Tier distribution bar chart (Low / Medium / High / Critical)
- Per-branch heatmap: Country × Room Type
- Independent drill-down selector → ranked bar chart of all matching segments
- Full risk table with gradient colouring

---

## Key Insights

1. **Branch matters most** — Changi (41.8% no-show) vs Orchard (27.7%). Same company, very different risk profile.
2. **China guests are the dominant risk driver** — 56.8% overall; 64.8% at Changi specifically.
3. **First-time guests are 2.6× more likely to no-show** — retention programmes directly cut no-show exposure.
4. **Booking channel has almost no effect** — Website, Agent, Email, Phone all cluster at 36–38%.
5. **CLV is heavily eroded by no-shows** — Changi/China CLV is less than half of Orchard/Japan despite similar room prices.

**Recommended actions:**
- Apply a deposit or overbooking buffer to Changi / China / King room bookings (Critical tier).
- Launch a loyalty programme targeting first-time guests — closing the first-time/returning gap alone would cut overall no-shows by ~8 pp.
- Prioritise revenue investment in the Orchard / Japan and Orchard / Australia segments for highest CLV return.
