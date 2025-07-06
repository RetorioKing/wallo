###############################################################################
# Full Analytics Dashboard (EDA • Regression • Classification • Clustering • APR)
# -- Tested on Streamlit 1.33  |  scikit-learn ≥ 1.2  |  pandas ≥ 2.0
###############################################################################

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             r2_score, mean_squared_error)
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

# ----------------------------  APP CONFIG  -----------------------------------
st.set_page_config(page_title="Full Analytics Dashboard", layout="wide")
st.title("📊 Full Analytics Dashboard")

# ----------------------------  FILE UPLOAD  ----------------------------------
uploaded_file = st.file_uploader("📤 Upload an Excel file", type=["xlsx"])

# ----------------------------------------------------------------------------- 
if uploaded_file:
    xls        = pd.ExcelFile(uploaded_file)
    sheet_name = st.selectbox("📄 Choose a sheet", xls.sheet_names)
    df         = xls.parse(sheet_name)
    st.success(f"Loaded **{sheet_name}** – shape: {df.shape}")

    # ------------------------------------------------------------------------- 
    #  TABS
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📈 Descriptive", "📉 Regression", "🎯 Classification", "👥 Clustering", "🔗 Assoc. Rules"]
    )

    # ----------------------  TAB 1: DESCRIPTIVE  -----------------------------
    with tab1:
        st.subheader("Summary Statistics")
        st.dataframe(df.describe())

        st.subheader("Correlation Heatmap")
        num_df = df.select_dtypes(include=np.number)
        if num_df.empty:
            st.info("No numeric columns available.")
        else:
            fig, ax = plt.subplots()
            sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

    # ----------------------  TAB 2: REGRESSION  ------------------------------
    with tab2:
        st.subheader("Linear • Ridge • Lasso Regression")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) < 2:
            st.info("Need at least 2 numeric columns for regression.")
        else:
            target   = st.selectbox("Target variable", numeric_cols)
            features = st.multiselect(
                "Predictor variables", [c for c in numeric_cols if c != target]
            )
            if features and st.button("Run Regression"):
                X, y      = df[features].copy(), df[target]
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                scaler           = StandardScaler()
                X_train_scaled   = scaler.fit_transform(X_train)
                X_test_scaled    = scaler.transform(X_test)

                models = {
                    "Linear" : LinearRegression(),
                    "Ridge"  : Ridge(),
                    "Lasso"  : Lasso()
                }

                for name, model in models.items():
                    model.fit(X_train_scaled, y_train)
                    pred  = model.predict(X_test_scaled)
                    st.markdown(f"**{name} Regression**")
                    st.write("• R²:", round(r2_score(y_test, pred), 3))
                    st.write("• RMSE:", round(np.sqrt(mean_squared_error(y_test, pred)), 3))

    # ----------------------  TAB 3: CLASSIFICATION  --------------------------
    with tab3:
        st.subheader("Random Forest (Binary Classification)")
        num_cols  = df.select_dtypes(include=np.number).columns.tolist()
        target_cl = st.selectbox("Target (binary 0/1)", num_cols, key="clf_tgt")
        feats_cl  = st.multiselect("Features", [c for c in num_cols if c != target_cl],
                                   key="clf_feats")

        if feats_cl and st.button("Run Classifier"):
            X, y = df[feats_cl], df[target_cl]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)

            clf = RandomForestClassifier(random_state=42)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))
            st.text("Confusion Matrix:")
            st.write(confusion_matrix(y_test, y_pred))

    # ----------------------  TAB 4: CLUSTERING  ------------------------------
    with tab4:
        st.subheader("K-Means Clustering")

        clust_feats = st.multiselect(
            "Select numeric features", df.select_dtypes(include=np.number).columns.tolist(),
            key="cluster_feats"
        )

        if len(clust_feats) >= 2:

            # --- Pre-processing ---
            X = df[clust_feats].copy()
            # Handle missing numerics
            X = X.fillna(X.mean(numeric_only=True))

            scaler  = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # --- Elbow chart ---
            st.markdown("##### Elbow Method")
            inertia = []
            max_k = min(10, len(X))  # can't have k > n_samples
            for k in range(1, max_k + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X_scaled)
                inertia.append(kmeans.inertia_)

            fig, ax = plt.subplots()
            ax.plot(range(1, max_k + 1), inertia, marker="o")
            ax.set_xlabel("k")
            ax.set_ylabel("Inertia")
            st.pyplot(fig)

            # --- Choose k ---
            k_val = st.slider("Choose k", 2, max_k, 3)

            # --- Fit final model ---
            try:
                kmeans         = KMeans(n_clusters=k_val, random_state=42, n_init=10)
                df["cluster"]  = kmeans.fit_predict(X_scaled)

                st.markdown("##### Cluster Profiles (mean values)")
                persona = df.groupby("cluster")[clust_feats].mean().round(2)
                st.dataframe(persona)

                # --- Download button ---
                towrite = io.BytesIO()
                with pd.ExcelWriter(towrite, engine="xlsxwriter") as writer:
                    df.to_excel(writer, index=False, sheet_name="Clustered_Data")
                towrite.seek(0)

                st.download_button(
                    label="💾 Download data with clusters",
                    data=towrite,
                    file_name="clustered_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.error(f"Clustering failed: {e}")

        else:
            st.info("Select **at least two** numeric features for clustering.")

    # ----------------------  TAB 5: ASSOCIATION RULES  -----------------------
    with tab5:
        st.subheader("Apriori – Association Rules")
        bool_df = df.select_dtypes(include="bool")
        if bool_df.empty:
            st.info("Need True/False columns to run Apriori.")
        else:
            support   = st.slider("Min support", 0.01, 1.0, 0.05, 0.01)
            confidence = st.slider("Min confidence", 0.10, 1.0, 0.3, 0.05)
            lift       = st.slider("Min lift", 1.0, 5.0, 1.0, 0.1)

            if st.button("Run Apriori"):
                freq_items = apriori(bool_df, min_support=support, use_colnames=True)
                if freq_items.empty:
                    st.warning("No frequent itemsets found with current support setting.")
                else:
                    rules = association_rules(freq_items, metric="confidence",
                                              min_threshold=confidence)
                    rules = rules[rules["lift"] >= lift]
                    if rules.empty:
                        st.warning("No rules matched confidence/lift thresholds.")
                    else:
                        st.dataframe(rules[["antecedents", "consequents",
                                            "support", "confidence", "lift"]])

# ----------------------------------------------------------------------------- 
else:
    st.info("⬆️ Upload an Excel file to begin.")
