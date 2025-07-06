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
from sklearn.metrics import classification_report, confusion_matrix, r2_score, mean_squared_error
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

# --- Config ---
st.set_page_config(page_title="Full Analytics Dashboard", layout="wide")
st.title("📊 Full Analytics Dashboard – EDA, ML, Clustering & More")

# --- Upload Excel File ---
uploaded_file = st.file_uploader("📤 Upload your Excel file", type=["xlsx"])

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    sheet_name = st.selectbox("📄 Select a sheet", xls.sheet_names)
    df = xls.parse(sheet_name)
    st.success(f"✅ Loaded {sheet_name} with shape {df.shape}")

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Descriptive Analytics", "📉 Regression", "🎯 Classification", "👥 Clustering", "🔗 Association Rules"
    ])

    # --- Tab 1: Descriptive Analytics ---
    with tab1:
        st.subheader("Summary Statistics")
        st.dataframe(df.describe())

        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # --- Tab 2: Regression ---
    with tab2:
        st.subheader("Regression (Linear / Ridge / Lasso)")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        target = st.selectbox("🎯 Select target variable", numeric_cols)
        features = st.multiselect("📌 Select independent variables", [col for col in numeric_cols if col != target])

        if features and st.button("Run Regression"):
            X = df[features]
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(),
                "Lasso Regression": Lasso()
            }

            for name, model in models.items():
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                st.subheader(name)
                st.write("R² Score:", round(r2_score(y_test, y_pred), 3))
                st.write("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 3))

    # --- Tab 3: Classification ---
    with tab3:
        st.subheader("Random Forest Classification")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        class_target = st.selectbox("🎯 Select classification target (binary)", num_cols)
        class_features = st.multiselect("📌 Select features for classification", [col for col in num_cols if col != class_target])

        if class_features and st.button("Run Classification"):
            X = df[class_features]
            y = df[class_target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))
            st.text("Confusion Matrix:")
            st.write(confusion_matrix(y_test, y_pred))

    # --- Tab 4: Clustering ---
    with tab4:
        st.subheader("K-Means Clustering")

        cluster_features = st.multiselect("🛠️ Select features for clustering", df.select_dtypes(include=np.number).columns.tolist())

        if len(cluster_features) >= 2:
            X = df[cluster_features]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Elbow Method
            st.subheader("📊 Elbow Chart")
            inertia = []
            for k in range(1, 11):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
                kmeans.fit(X_scaled)
                inertia.append(kmeans.inertia_)

            fig, ax = plt.subplots()
            ax.plot(range(1, 11), inertia, marker='o')
            ax.set_title("Elbow Method for Optimal Clusters")
            ax.set_xlabel("k")
            ax.set_ylabel("Inertia")
            st.pyplot(fig)

            # Cluster slider
            k_val = st.slider("🔢 Select number of clusters", 2, 10, 3)

            # Run KMeans
            kmeans = KMeans(n_clusters=k_val, random_state=42, n_init='auto')
            df['cluster'] = kmeans.fit_predict(X_scaled)

            st.subheader("👤 Cluster Personas (Averages)")
            persona = df.groupby('cluster')[cluster_features].mean().round(2)
            st.dataframe(persona)

            # Download
            st.subheader("📥 Download Clustered Dataset")
            towrite = io.BytesIO()
            with pd.ExcelWriter(towrite, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Clustered_Data')
            towrite.seek(0)

            st.download_button(
                label="Download Excel with Clusters",
                data=towrite,
                file_name="clustered_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.warning("⚠️ Please select at least 2 numeric features for clustering.")

    # --- Tab 5: Association Rules ---
    with tab5:
        st.subheader("Market Basket – Association Rule Mining")
        if df.select_dtypes(include='bool').shape[1] > 0:
            min_support = st.slider("📊 Minimum Support", 0.01, 1.0, 0.1)
            try:
                frequent_items = apriori(df.select_dtypes(include='bool'), min_support=min_support, use_colnames=True)
                rules = association_rules(frequent_items, metric="lift", min_threshold=1.0)
                st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
            except Exception as e:
                st.error("Error running Apriori. Please ensure binary (True/False) format.")
        else:
            st.warning("⚠️ No binary columns available for Apriori.")
else:
    st.info("⬆️ Upload an Excel file to begin.")
