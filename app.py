###############################################################################
# Quick Analytics Dashboard  –  Descriptive • Regression • Elbow Insight
# Streamlit ≥1.33  |  scikit-learn ≥1.2  |  pandas ≥2.0
###############################################################################
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ───────────────────────────  CONFIG  ───────────────────────────────────────
st.set_page_config(page_title="Quick Analytics Dashboard", layout="wide")
st.title("📊 Quick Analytics Dashboard")

# ───────────────────────── 1) LOAD DATA  ────────────────────────────────────
uploaded = st.sidebar.file_uploader("📤 Upload an Excel file", type=["xlsx"])
if not uploaded:
    st.info("⬅️ Please upload a workbook to begin.")
    st.stop()

sheet = st.sidebar.selectbox("🗂 Select sheet", pd.ExcelFile(uploaded).sheet_names)
df    = pd.read_excel(uploaded, sheet_name=sheet)
st.success(f"Loaded **{sheet}** – {df.shape[0]:,} rows × {df.shape[1]} columns")
st.dataframe(df.head())

# ───────────────────────── 2) BUILD TABS  ───────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 Descriptive Stats", "📉 Regression", "👥 Elbow Insight"])

# ───────────────────────  TAB 1 – DESCRIPTIVE  ─────────────────────────────
with tab1:
    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

# ───────────────────────  TAB 2 – REGRESSION   ─────────────────────────────
with tab2:
    st.subheader("Simple Linear Regression Benchmark")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("Need at least two numeric columns.")
    else:
        target   = st.selectbox("Target variable", numeric_cols, key="reg_target")
        features = [c for c in numeric_cols if c != target]

        X = df[features].fillna(df[features].mean())
        y = df[target]
        X_scaled = StandardScaler().fit_transform(X)

        from sklearn.linear_model import LinearRegression
        model  = LinearRegression().fit(X_scaled, y)
        r2_all = model.score(X_scaled, y)
        st.write(f"R² on full data: **{r2_all:.3f}**")

# ───────────────────────  TAB 3 – ELBOW CHART  ─────────────────────────────
with tab3:
    st.subheader("Elbow Method – Determining Optimal k")

    # Feature choice
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    chosen   = st.multiselect("Numeric columns used for clustering distances",
                              num_cols, default=num_cols)

    if len(chosen) < 2:
        st.warning("Select at least two numeric columns.")
        st.stop()

    # Standardize
    X = df[chosen].fillna(df[chosen].mean())
    X = StandardScaler().fit_transform(X)

    # Compute inertia for k = 2–10
    k_range = range(2, 11)
    inertia = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertia.append(km.inertia_)

    # Plot elbow curve
    fig, ax = plt.subplots()
    ax.plot(k_range, inertia, marker="o")
    ax.set_title("Elbow Method")
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("Inertia")
    st.pyplot(fig)

    # Build explanatory table
    pct_drop = np.diff(inertia) / inertia[:-1] * -100  # positive % drops
    elbow_k  = None
    for k_idx, pct in enumerate(pct_drop[1:], start=3):  # start checking at k=3
        if pct < 15:                                     # small drop ⇒ elbow
            elbow_k = k_idx
            break
    elbow_k = elbow_k or 3  # default to 3 if not found

    explain_tbl = pd.DataFrame({
        "k": list(k_range),
        "Inertia": [f"{val:,.0f}" for val in inertia],
        "Δ% vs prev": ["–"] + [f"{d:.1f}%" for d in pct_drop],
        "Comment": ["Start"] + [
            "Elbow candidate" if k == elbow_k else
            ("Sharp drop" if d > 30 else "Diminishing returns")
            for k, d in zip(list(k_range)[1:], pct_drop)
        ]
    })
    st.dataframe(explain_tbl, use_container_width=True)

    st.markdown(
        f"""
**Interpretation:**  
* Inertia measures how compact the clusters are – lower is better.  
* Notice large drops until **k ≈ {elbow_k}**, then improvements level off.  
* Therefore, **k = {elbow_k}** is a sensible choice; adding more clusters after that offers minimal benefit.
"""
    )
