# Model Card — Banking Churn Prediction

**Version:** 1.0  
**Date:** 2026-05-08  
**Author:** Nicolás Zuleta Sierra  
**Contact:** nicolaszuleta95@gmail.com  

---

## Model Overview

| Property | Details |
|---|---|
| **Model type** | XGBoost Classifier (gradient boosted trees) |
| **Task** | Binary classification — customer churn prediction |
| **Input** | 13 customer features (demographic, behavioral, product) |
| **Output** | Churn probability ∈ [0, 1] + binary label at tuned threshold |
| **Threshold** | 0.554 (tuned to maximize Recall with Precision ≥ 75%) |
| **Serialization** | Single `.joblib` artifact containing full sklearn pipeline |

---

## Intended Use

### Primary use cases
- **Proactive retention:** Identify customers at risk of churning 30–45 days before predicted departure so retention teams can intervene
- **Prioritization:** Rank customers by churn probability to allocate finite retention resources efficiently
- **Business impact quantification:** Estimate revenue at risk and ROI of retention programs

### Out-of-scope uses
- **Real-time scoring at scale** without load testing and latency optimization
- **Automated account actions** (freezes, product changes) without human review
- **Markets outside France, Germany, Spain** — the model has not been validated on other geographies
- **Credit decisions** — this model predicts churn, not creditworthiness

---

## Training Data

| Property | Value |
|---|---|
| **Source** | [Bank Customer Churn Prediction — Kaggle](https://www.kaggle.com/datasets/shubhammeshram579/bank-customer-churn-prediction) |
| **Records** | 10,000 customers |
| **Train split** | 8,000 (80%) — stratified |
| **Test split** | 2,000 (20%) — stratified, hold-out |
| **Churn rate** | 20.4% (class ratio ~4:1) |
| **Imbalance handling** | SMOTE on training data only (k_neighbors=5) |
| **Time period** | Not disclosed in source dataset |

### Features used (13)

| Feature | Type | Notes |
|---|---|---|
| `CreditScore` | Numeric | Clipped to [300, 850] |
| `Geography` | Categorical | France / Germany / Spain |
| `Gender` | Categorical | Female / Male |
| `Age` | Numeric | Clipped to [18, 100] |
| `Tenure` | Numeric | Years as customer |
| `Balance` | Numeric | Account balance in EUR |
| `NumOfProducts` | Numeric | Number of bank products held |
| `HasCrCard` | Binary | Owns a credit card |
| `IsActiveMember` | Binary | Active member flag |
| `EstimatedSalary` | Numeric | Annual salary estimate |
| `Satisfaction Score` | Ordinal (1–5) | Customer satisfaction survey result |
| `Card Type` | Categorical | SILVER / GOLD / PLATINUM / DIAMOND |
| `Point Earned` | Numeric | Loyalty program points |

### Engineered features (7)

| Feature | Description | Motivation from EDA |
|---|---|---|
| `age_segment` | young / mid / senior / elder | Seniors (51–60) churn 3× more than young customers |
| `zero_balance` | 1 if Balance = 0 | Zero-balance customers have lower churn (13.8% vs 24.1%) — counterintuitive |
| `balance_per_product` | Balance / NumOfProducts | Financial engagement ratio |
| `products_3plus` | 1 if NumOfProducts ≥ 3 | Products 3–4 yield 82–100% churn rate |
| `is_german` | 1 if Geography = Germany | Germany churn 32% vs France/Spain ~16% |
| `engagement_score` | IsActiveMember × (1 − zero_balance) × log(Tenure+1) | Composite activity signal |
| `risk_profile` | Combined balance + products + activity indicator | Holistic risk scoring |

---

## Performance Metrics

Evaluated on the 20% hold-out test set (2,000 customers, 408 churners):

| Metric | Value | Business Interpretation |
|---|---|---|
| **AUC-ROC** | **0.8726** | Strong discriminative power |
| **Precision** | **0.7526** | 75% of flagged customers are genuinely at risk |
| **Recall** | **0.5294** | Identifies 53% of customers who will churn |
| **F1-Score** | **0.6216** | Balanced performance |
| **Accuracy** | **0.8685** | Overall prediction accuracy |
| **CV AUC (5-fold)** | **0.8574 ± 0.0071** | Stable generalization |

### Why Recall is 53%

This model excludes the `Complain` feature (see Data Leakage section below). Without it, the precision-recall tradeoff on this dataset limits achievable recall at Precision ≥ 75%. A model including `Complain` achieves AUC ~0.999 but with strong data leakage concerns.

---

## Data Leakage — Critical Note on `Complain`

The dataset contains a `Complain` feature with a **99.5% correlation with churn**:
- Customers with `Complain = 1` → churn rate 99.5%
- Customers with `Complain = 0` → churn rate 0.1%

This extreme correlation strongly suggests that complaints in this dataset are **recorded concurrently with or after the churn event**, not as a genuine leading indicator. Including `Complain` in the model:

- Inflates AUC to ~0.999 (essentially memorizing the target)
- Would yield misleading performance estimates
- Would not generalize to production where complaints precede churn by unknown time

**Decision:** `Complain` is excluded from production features. If your operational system records complaints reliably *before* churn occurs (e.g., via CRM with timestamps), re-introducing `Complain` with proper temporal validation could significantly improve model performance.

---

## Model Architecture

```
Input DataFrame (13 features)
    │
    ▼
ChurnFeatureEngineer          # 7 derived features added
    │
    ▼
ColumnTransformer
  ├── StandardScaler          # numeric features
  ├── passthrough             # binary/ordinal features
  ├── OneHotEncoder           # Geography, Gender, age_segment
  └── OrdinalEncoder          # Card Type (SILVER < GOLD < PLATINUM < DIAMOND)
    │
    ▼
SMOTE (training only)         # Oversample minority class 1:1
    │
    ▼
XGBClassifier
  n_estimators=500, max_depth=6, learning_rate=0.03
  subsample=0.85, colsample_bytree=0.85, min_child_weight=3
    │
    ▼
Threshold at 0.554 → Binary Label
```

---

## Key Drivers of Churn (SHAP)

Based on SHAP TreeExplainer analysis on the test set:

1. **Age** — Older customers (51+) are at significantly higher risk
2. **NumOfProducts** — 3–4 products → extreme churn (82–100%)
3. **IsActiveMember** — Inactive members churn 2× more than active ones
4. **Balance** — High non-zero balances correlate with churn in this dataset
5. **Geography** — German customers churn at 2× the rate of French/Spanish customers
6. **products_3plus** (engineered) — Binary flag capturing the 3–4 product extreme
7. **Satisfaction Score** — Some signal, though not monotonically linear

---

## Known Limitations

1. **Dataset origin:** Synthetic/anonymized Kaggle dataset. Real-world distributions may differ significantly.
2. **Temporal validity:** No timestamps available. The model cannot account for seasonality or market events.
3. **Geographic scope:** Only validated for France, Germany, Spain. Predictions for other markets are unreliable.
4. **Class imbalance:** Despite SMOTE, rare event prediction is inherently uncertain at low churn rates.
5. **Static features:** The model uses point-in-time snapshots. Dynamic features (e.g., recent transaction velocity) could improve performance.
6. **Recall constraint:** At Precision ≥ 75%, recall is ~53%. 47% of true churners are missed. Intervention programs should account for this.

---

## Fairness Considerations

Preliminary analysis of churn rates across demographic groups:

| Subgroup | Churn Rate | Vs. Overall (20.4%) |
|---|---|---|
| Female | 25.1% | +4.7pp |
| Male | 16.5% | −3.9pp |
| Germany | 32.4% | +12pp |
| France | 16.2% | −4.2pp |
| Spain | 16.7% | −3.7pp |
| Age 51–60 | ~45% | +25pp |
| Age 18–35 | ~10% | −10pp |

**Recommendation:** Before deployment, audit whether model predictions result in systematically different retention offer rates for these demographic groups. Ensure retention interventions are equitable and not inadvertently used for discriminatory decisions.

---

## Deployment Recommendations

1. **Validate temporally:** Re-train with a proper temporal train/test split (e.g., train on customers from Q1–Q3, test on Q4) to avoid look-ahead bias.
2. **Monitor distribution shift:** Track feature distributions and model calibration monthly. Retrain if AUC drops >3pp.
3. **A/B test the intervention:** Compare retention rates for model-flagged customers who received outreach vs. a control group.
4. **Set intervention thresholds by segment:** Consider different thresholds for high-CLV vs. low-CLV customers.
5. **Human-in-the-loop:** Do not automate product changes or account actions based solely on model output.

---

## Ethical Considerations

- This model should not be used to deny services or apply punitive measures to customers predicted to churn
- Retention interventions should be genuinely beneficial to customers (e.g., better rates, personalized offers), not manipulative
- Model predictions should not be shared externally or used outside of the intended retention use case

---

*This model card follows the [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) framework (Mitchell et al., 2019).*
