import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

# App config
st.set_page_config(page_title="Walmart Analytics Dashboard", layout="wide")
st.title("📊 Walmart Analytics Dashboard")

# Upload Excel
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
if uploaded_file:
    # Load workbook
    xls = pd.ExcelFile(uploaded_file)
    sheet_name = st.selectbox("Select sheet", xls.sheet_names)
    df = xls.parse(sheet_name)

    st.success(f"Loaded dataset: {sheet_name} with shape {df.shape}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Descriptive Analytics", "Regression", "Classification", "Clustering", "Association Rules"
    ])

    # Tab 1: Descriptive Analytics
    with tab1:
        st.header("📈 Descriptive Analytics")
        st.dataframe(df.head())
        st.subheader("Summary Statistics")
        st.write(df.describe())
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    # Tab 2: Regression
    with tab2:
        st.header("📉 Regression Modeling")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        target = st.selectbox("Select target variable", numeric_cols)
        features = st.multiselect("Select independent variables", [col for col in numeric_cols if col != target])

        if st.button("Run Regression") and features:
            X = df[features]
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            models = {
                "Linear": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso()
            }

            for name, model in models.items():
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                st.subheader(f"{name} Regression")
                st.write("R² Score:", r2_score(y_test, y_pred))
                st.write("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

    # Tab 3: Classification
    with tab3:
        st.header("🎯 Classification Modeling")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        class_target = st.selectbox("Select classification target (binary)", numeric_cols)
        class_features = st.multiselect("Select features", [col for col in numeric_cols if col != class_target])

        if st.button("Run Classifier") and class_features:
            X = df[class_features]
            y = df[class_target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            clf = RandomForestClassifier(random_state=42)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            st.subheader("Random Forest Classification Results")
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))
            st.text("Confusion Matrix:")
            st.write(confusion_matrix(y_test, y_pred))

    # Tab 4: Clustering
    with tab4:
        st.header("👥 K-Means Clustering")
        clustering_features = st.multiselect("Select features for clustering", df.select_dtypes(include=np.number).columns.tolist())
        if len(clustering_features) >= 2:
            X = df[clustering_features]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Elbow Chart
            st.subheader("📊 Elbow Chart to Choose k")
            inertia = []
            K = range(1, 11)
            for k in K:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(X_scaled)
                inertia.append(kmeans.inertia_)
            fig, ax = plt.subplots()
            ax.plot(K, inertia, marker='o')
            ax.set_xlabel('k')
            ax.set_ylabel('Inertia')
            ax.set_title('Elbow Method')
            st.pyplot(fig)

            # Slider to select k
            num_clusters = st.slider("Select number of clusters", 2, 10, 3)
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            df['cluster'] = kmeans.fit_predict(X_scaled)

            # Cluster Persona Table
            st.subheader("🧠 Cluster Persona Table")
            persona = df.groupby('cluster')[clustering_features].mean().round(2)
            st.dataframe(persona)

            # Download button
            st.subheader("📥 Download Clustered Data")
            towrite = io.BytesIO()
            with pd.ExcelWriter(towrite, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Clustered_Data')
            towrite.seek(0)

            st.download_button(
                label="Download Excel with Cluster Labels",
                data=towrite,
                file_name="clustered_output.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.warning("Please select at least 2 features for clustering.")

    # Tab 5: Association Rules
    with tab5:
        st.header("🔗 Association Rule Mining")
        if df.select_dtypes(include='object').shape[1] > 0:
            st.write("Make sure your data is in one-hot encoded format for this to work.")
            min_support = st.slider("Minimum support", 0.01, 1.0, 0.1)
            try:
                freq_items = apriori(df.select_dtypes(include='bool'), min_support=min_support, use_colnames=True)
                rules = association_rules(freq_items, metric="lift", min_threshold=1.0)
                st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
            except:
                st.warning("Please ensure your data is binary encoded (True/False) for Apriori to work.")
        else:
            st.warning("No object/categorical columns to use for association rules.")

else:
    st.info("Please upload an Excel file to get started.")
