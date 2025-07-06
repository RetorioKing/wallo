###############################################################################
# Ultra-Fast Clustering Dashboard
# Tested on: Streamlit ≥1.33  |  scikit-learn ≥1.2  |  pandas ≥2.0
###############################################################################
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans

# ────────────────────────  STREAMLIT CONFIG  ────────────────────────────────
st.set_page_config(page_title="Ultra-Fast Clustering Dashboard", layout="wide")
st.title("🧠 Ultra-Fast Clustering & EDA")

# ───────────────────────── 1) DATA SOURCE PICKER ────────────────────────────
@st.cache_data(show_spinner=False)
def read_excel(upload, sheet_name):
    """Cache Excel sheet loading to speed up repeated runs."""
    return pd.read_excel(upload, sheet_name=sheet_name)

uploaded = st.sidebar.file_uploader(
    "📤 Upload an Excel workbook (optional)", type=["xlsx"]
)

# Demo (synthetic) datasets already in session from your earlier code
built_in = {
    "Cart Abandonment":    st.session_state.get("Cart_Abandonment_Dataset"),
    "Warehousing":         st.session_state.get("Warehousing_Optimization_Dataset"),
    "Review Authenticity": st.session_state.get("Review_Authenticity_Dataset"),
}

if uploaded:
    sheet = st.sidebar.selectbox(
        "🗂 Choose a sheet", pd.ExcelFile(uploaded).sheet_names
    )
    if sheet:
        df = read_excel(uploaded, sheet)
        data_label = f"{uploaded.name} › {sheet}"
else:
    demo_choice = st.sidebar.selectbox("🗂 Pick a demo dataset", list(built_in.keys()))
    df = built_in[demo_choice]
    data_label = demo_choice

if df is None:
    st.info("⬅️ Upload a file or choose a demo set in the sidebar.")
    st.stop()

st.success(f"Loaded **{data_label}** — {df.shape[0]:,} rows × {df.shape[1]} cols")
st.dataframe(df.head())

# ──────────────────────── 2) FEATURE SELECTION ──────────────────────────────
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
st.subheader("Step 1 – Select numeric features for clustering")
features = st.multiselect("Numeric columns", numeric_cols, default=numeric_cols)

if len(features) < 2:
    st.warning("Please select at least two numeric columns.")
    st.stop()

# ───────────────────── 3) MINI-BATCH K-MEANS: ELBOW & FIT ───────────────────
X = df[features].fillna(df[features].mean(numeric_only=True))
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

max_possible_k = min(10, len(X_scaled))        # cannot exceed sample size
elbow_limit   = st.slider(
    "Step 2 – Preview elbow curve up to k", 2, max_possible_k, 6
)

progress = st.progress(0, text="Calculating elbow curve …")
inertia_vals = []
for k in range(1, elbow_limit + 1):
    mbk = MiniBatchKMeans(
        n_clusters=k, random_state=42, n_init=5, batch_size=1024
    )
    mbk.fit(X_scaled)
    inertia_vals.append(mbk.inertia_)
    progress.progress(k / elbow_limit)
progress.empty()

fig, ax = plt.subplots()
ax.plot(range(1, elbow_limit + 1), inertia_vals, marker="o")
ax.set_xlabel("k")
ax.set_ylabel("Inertia")
ax.set_title("Elbow Method")
st.pyplot(fig)

k_final = st.slider(
    "Step 3 – Choose final number of clusters", 2, elbow_limit, min(3, elbow_limit)
)
run_cluster = st.button("🚀 Run MiniBatch K-Means")

# ─────────────────────────── 4) CLUSTER & OUTPUT ────────────────────────────
if run_cluster:
    with st.spinner("Clustering …"):
        mbk_final = MiniBatchKMeans(
            n_clusters=k_final, random_state=42, n_init=10, batch_size=1024
        )
        df["cluster"] = mbk_final.fit_predict(X_scaled)

    st.success("Clustering complete!")
    st.subheader("Cluster profiles (mean values)")
    st.dataframe(df.groupby("cluster")[features].mean().round(2))

    # Excel download
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Clustered_Data")
    buffer.seek(0)

    st.download_button(
        "💾 Download Excel with clusters",
        data=buffer,
        file_name="clustered_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
