"""
Streamlit dashboard — Banking Churn Prediction
Three sections:
  1. Model Overview  — metrics, ROC, score distributions
  2. Individual Prediction — customer profile → churn probability + SHAP
  3. Business Impact Dashboard — revenue at risk, ROI of retention
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import shap
import streamlit as st
import joblib
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

from data_processing import load_raw, basic_cleaning, split_features_target, FEATURE_COLS
from features import ChurnFeatureEngineer
from predict import load_model, predict_proba, predict_single, explain, business_impact

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Banking Churn Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
DATA_PATH = ROOT / "data" / "raw" / "Customer-Churn-Records.csv"


# ── Cached loaders ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_artifact():
    try:
        return load_model(MODEL_PATH), None
    except FileNotFoundError as e:
        return None, str(e)


@st.cache_data
def get_test_data():
    df = load_raw(DATA_PATH)
    df = basic_cleaning(df)
    X, y = split_features_target(df)
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    return X_test.reset_index(drop=True), y_test.reset_index(drop=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.shields.io/badge/XGBoost-Powered-FF6600?style=flat", width=200)
st.sidebar.title("🏦 Churn Prediction")
st.sidebar.markdown("**End-to-end ML system** for retail banking customer churn.")

section = st.sidebar.radio(
    "Navigate",
    ["📊 Model Overview", "🔍 Individual Prediction", "💼 Business Impact"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** 10,000 customers · 20.4% churn rate")
st.sidebar.markdown("**Model:** XGBoost + SMOTE")
st.sidebar.markdown("**Author:** Nicolás Zuleta Sierra")

# ── Load artifact ─────────────────────────────────────────────────────────────
artifact, load_error = get_artifact()

if load_error:
    st.error(f"⚠️ Model not found. {load_error}")
    st.info("Run `python src/train.py` from the project root to train the model first.")
    st.stop()

threshold = artifact["threshold"]
metrics = artifact["metrics"]

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Model Overview
# ═══════════════════════════════════════════════════════════════════════════════
if section == "📊 Model Overview":
    st.title("📊 Model Overview")
    st.markdown("Performance of the final XGBoost model on the 20% hold-out test set.")

    # Metric cards
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("AUC-ROC", f"{metrics['auc_roc']:.4f}", help="Area under the ROC curve")
    col2.metric("Precision", f"{metrics['precision']:.4f}", help="Of flagged customers, % truly at risk")
    col3.metric("Recall", f"{metrics['recall']:.4f}", help="% of churners correctly identified")
    col4.metric("F1-Score", f"{metrics['f1']:.4f}", help="Harmonic mean of Precision & Recall")
    col5.metric("Threshold", f"{threshold}", help="Classification threshold (tuned for Precision ≥ 0.75)")

    st.markdown(f"**CV AUC (5-fold):** {artifact['cv_auc_mean']:.4f} ± {artifact['cv_auc_std']:.4f}")
    st.divider()

    # ROC + Score distribution
    X_test, y_test = get_test_data()
    y_proba = predict_proba(artifact, X_test)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("ROC Curve")
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, color="#DD8452", lw=2, label=f"XGBoost (AUC={auc:.4f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
        ax.fill_between(fpr, tpr, alpha=0.08, color="#DD8452")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve — Test Set")
        ax.legend(loc="lower right")
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.subheader("Score Distribution")
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.hist(y_proba[y_test == 0], bins=50, alpha=0.6, color="#4C72B0",
                label="No Churn", density=True)
        ax.hist(y_proba[y_test == 1], bins=50, alpha=0.6, color="#DD8452",
                label="Churn", density=True)
        ax.axvline(threshold, color="red", linestyle="--", lw=2, label=f"Threshold={threshold}")
        ax.set_xlabel("Churn Probability")
        ax.set_ylabel("Density")
        ax.set_title("Predicted Probability Distribution")
        ax.legend()
        st.pyplot(fig)
        plt.close()

    # Confusion matrix
    st.subheader("Confusion Matrix")
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"])
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    ax.set_title(f"Confusion Matrix (threshold={threshold})")
    col_cm, _ = st.columns([1, 1])
    with col_cm:
        st.pyplot(fig)
    plt.close()

    st.info(
        f"**Why this threshold?** At threshold={threshold}, the model achieves "
        f"Precision={metrics['precision']:.2%} and Recall={metrics['recall']:.2%}. "
        "In a churn context, missing an at-risk customer is more costly than a false alarm. "
        "The threshold was tuned to maximize Recall while keeping Precision ≥ 75%."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Individual Prediction
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "🔍 Individual Prediction":
    st.title("🔍 Individual Customer Prediction")
    st.markdown("Enter a customer profile to get their churn probability and a SHAP explanation.")

    with st.form("customer_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Demographics**")
            age = st.slider("Age", 18, 92, 45)
            gender = st.selectbox("Gender", ["Female", "Male"])
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"])

        with col2:
            st.markdown("**Banking Profile**")
            credit_score = st.slider("Credit Score", 300, 850, 650)
            tenure = st.slider("Tenure (years)", 0, 10, 3)
            balance = st.number_input("Balance ($)", 0.0, 300000.0, 75000.0, step=1000.0)
            num_products = st.selectbox("# Products", [1, 2, 3, 4])

        with col3:
            st.markdown("**Activity & Perks**")
            has_cr_card = st.selectbox("Has Credit Card", [1, 0], format_func=lambda x: "Yes" if x else "No")
            is_active = st.selectbox("Is Active Member", [1, 0], format_func=lambda x: "Yes" if x else "No")
            salary = st.number_input("Estimated Salary ($)", 10000.0, 200000.0, 85000.0, step=1000.0)
            satisfaction = st.slider("Satisfaction Score (1-5)", 1, 5, 3)
            card_type = st.selectbox("Card Type", ["SILVER", "GOLD", "PLATINUM", "DIAMOND"])
            points = st.slider("Points Earned", 100, 1000, 400)

        submitted = st.form_submit_button("🔮 Predict Churn Risk", use_container_width=True)

    if submitted:
        customer = {
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": has_cr_card,
            "IsActiveMember": is_active,
            "EstimatedSalary": salary,
            "Satisfaction Score": satisfaction,
            "Card Type": card_type,
            "Point Earned": points,
        }

        result = predict_single(artifact, customer)
        prob = result["probability"]
        risk = result["risk_level"]

        color_map = {"Low": "🟢", "Medium": "🟡", "High": "🟠", "Very High": "🔴"}
        emoji = color_map.get(risk, "⚪")

        st.divider()
        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.metric("Churn Probability", f"{prob:.1%}")
        col_res2.metric("Risk Level", f"{emoji} {risk}")
        col_res3.metric("Decision", "⚠️ Flag for Retention" if result["label"] == 1 else "✅ Not at Risk")

        if result["label"] == 1:
            st.warning(f"This customer has a **{prob:.1%}** probability of churning — above the threshold ({threshold}). Recommend proactive retention contact.")
        else:
            st.success(f"This customer has a **{prob:.1%}** probability of churning — below the threshold ({threshold}). No immediate action needed.")

        # SHAP explanation
        st.subheader("Why? — SHAP Feature Explanation")
        with st.spinner("Computing SHAP values..."):
            try:
                shap_result = explain(artifact, customer, max_display=12)
                shap_vals = shap_result["shap_values"]
                feat_names = shap_result["feature_names"]

                fig, ax = plt.subplots(figsize=(9, 5))
                colors = ["#DD8452" if v > 0 else "#4C72B0" for v in shap_vals]
                y_pos = range(len(shap_vals))
                ax.barh(list(y_pos), shap_vals[::-1], color=colors[::-1], edgecolor="white")
                ax.set_yticks(list(y_pos))
                ax.set_yticklabels(feat_names[::-1])
                ax.axvline(0, color="gray", linewidth=0.8)
                ax.set_xlabel("SHAP Value (impact on churn log-odds)")
                ax.set_title(f"Feature Contributions — P(Churn) = {prob:.3f}")
                st.pyplot(fig)
                plt.close()

                st.caption("🟠 Orange = increases churn risk · 🔵 Blue = decreases churn risk")
            except Exception as e:
                st.warning(f"SHAP explanation unavailable: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Business Impact Dashboard
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "💼 Business Impact":
    st.title("💼 Business Impact Dashboard")
    st.markdown("Project the financial impact of deploying this model on your customer base.")

    st.subheader("Business Assumptions")
    col1, col2, col3 = st.columns(3)
    with col1:
        total_customers = st.number_input("Total Customers", 1000, 10_000_000, 50_000, step=1000)
        clv = st.number_input("Customer Lifetime Value ($)", 100, 10000, 800, step=50)
    with col2:
        contact_cost = st.number_input("Cost per Retention Contact ($)", 5, 500, 15, step=5)
        retention_rate = st.slider("Retention Success Rate (%)", 5, 60, 30) / 100
    with col3:
        st.markdown("")
        st.markdown("")
        run_impact = st.button("📊 Calculate Impact", use_container_width=True)

    st.divider()

    # Always show impact based on test set scaled to total_customers
    X_test, y_test = get_test_data()
    impact = business_impact(
        artifact, X_test,
        clv=clv, contact_cost=contact_cost, retention_rate=retention_rate,
    )

    # Scale from test set (2000) to total_customers
    scale = total_customers / impact["total_customers"]

    scaled = {
        "predicted_churners": int(impact["predicted_churners"] * scale),
        "revenue_at_risk": impact["revenue_at_risk"] * scale,
        "recoverable_revenue": impact["recoverable_revenue"] * scale,
        "intervention_cost": impact["intervention_cost"] * scale,
        "net_value": impact["net_value"] * scale,
    }

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers Flagged", f"{scaled['predicted_churners']:,}",
                f"{impact['churn_rate_predicted']:.1%} of base")
    col2.metric("Revenue at Risk", f"${scaled['revenue_at_risk']:,.0f}")
    col3.metric("Recoverable Revenue", f"${scaled['recoverable_revenue']:,.0f}",
                f"at {retention_rate:.0%} retention")
    col4.metric("Net Value (after costs)", f"${scaled['net_value']:,.0f}",
                f"ROI: {scaled['net_value']/max(scaled['intervention_cost'],1):.1f}×")

    st.divider()

    # Waterfall chart
    st.subheader("Value Waterfall")
    fig, ax = plt.subplots(figsize=(10, 4))
    categories = ["Revenue at Risk", "Recoverable\n(Ret. Rate)", "Intervention\nCost", "Net Value"]
    values = [
        scaled["revenue_at_risk"],
        scaled["recoverable_revenue"],
        -scaled["intervention_cost"],
        scaled["net_value"],
    ]
    colors = ["#4C72B0", "#55A868", "#DD8452", "#2ecc71"]
    bars = ax.bar(categories, [abs(v) for v in values], color=colors, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, values):
        label = f"${abs(val):,.0f}" if abs(val) < 1e6 else f"${abs(val)/1e6:.1f}M"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                label, ha="center", fontweight="bold", fontsize=9)
    ax.set_ylabel("USD")
    ax.set_title(f"Business Impact — {total_customers:,} Customers", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"${x/1e6:.1f}M" if x >= 1e6 else f"${x:,.0f}"
    ))
    st.pyplot(fig)
    plt.close()

    # Sensitivity table
    st.subheader("Sensitivity Analysis — Retention Rate")
    rows = []
    for ret in [0.10, 0.20, 0.30, 0.40, 0.50]:
        imp = business_impact(artifact, X_test, clv=clv, contact_cost=contact_cost, retention_rate=ret)
        rows.append({
            "Retention Rate": f"{ret:.0%}",
            "Recoverable Revenue": f"${imp['recoverable_revenue'] * scale:,.0f}",
            "Intervention Cost": f"${imp['intervention_cost'] * scale:,.0f}",
            "Net Value": f"${imp['net_value'] * scale:,.0f}",
            "ROI": f"{imp['net_value']/max(imp['intervention_cost'],1):.1f}×",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.info(
        "**Methodology:** The model flags customers with P(Churn) ≥ "
        f"{threshold}. Revenue figures assume CLV is fully realized for retained customers. "
        "Results scale linearly from the 2,000-customer hold-out test set."
    )
