import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from kmodes.kprototypes import KPrototypes
from mlxtend.frequent_patterns import apriori, association_rules
import os

# PAGE CONFIG
st.set_page_config(page_title="Portfolio Analytics Dashboard", page_icon="📈", layout="wide")

# SESSION STATE for file and sheet selection
if "active_excel_file" not in st.session_state:
    st.session_state.active_excel_file = None
if "active_sheet" not in st.session_state:
    st.session_state.active_sheet = None

# List available Excel files in root directory
def get_excel_files():
    return [f for f in os.listdir() if f.lower().endswith('.xlsx')]

# Get all sheet names from a given Excel file
def get_sheet_names(file):
    try:
        return pd.ExcelFile(file).sheet_names
    except Exception:
        return []

# TABS
tabs = [
    "📄 View & Switch Dataset",
    "📊 Descriptive Analytics",
    "🤖 Classification",
    "🎯 Clustering (K-Prototypes)",
    "🛒 Association Rules",
    "📈 Regression"
]
page = st.sidebar.radio("Choose analytics module", tabs)

# TAB 1: Dataset selection and preview
if page == "📄 View & Switch Dataset":
    st.header("📄 View & Switch Dataset")
    # Upload file UI
    uploaded_file = st.file_uploader("Upload Excel file", type="xlsx")
    if uploaded_file:
        file_path = uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    excel_files = get_excel_files()
    if not excel_files:
        st.info("Please upload an Excel (.xlsx) file to begin.")
        st.stop()
    # File select
    selected_file = st.selectbox("Select Excel file", excel_files, 
                                 index=excel_files.index(st.session_state.active_excel_file) 
                                 if st.session_state.active_excel_file in excel_files else 0)
    # Sheet select
    sheets = get_sheet_names(selected_file)
    if not sheets:
        st.warning("No sheets found in this file.")
        st.stop()
    selected_sheet = st.selectbox("Select sheet", sheets, 
                                  index=sheets.index(st.session_state.active_sheet) 
                                  if st.session_state.active_sheet in sheets else 0)

    # Button to activate dataset for analytics tabs
    if st.button("Set as active dataset for analytics"):
        st.session_state.active_excel_file = selected_file
        st.session_state.active_sheet = selected_sheet
        st.success(f"Set '{selected_file}' → sheet '{selected_sheet}' as active dataset for analytics.")

    # Preview dataframe
    try:
        df_preview = pd.read_excel(selected_file, sheet_name=selected_sheet)
        st.dataframe(df_preview, use_container_width=True)
        st.info(f"Previewing: '{selected_file}' → '{selected_sheet}' ({df_preview.shape[0]} rows, {df_preview.shape[1]} columns)")
    except Exception as e:
        st.error(f"Error loading sheet: {e}")

# OTHER TABS: Use remembered file+sheet
else:
    file = st.session_state.active_excel_file
    sheet = st.session_state.active_sheet
    if file is None or sheet is None:
        st.warning("Please choose an Excel file and sheet in the 'View & Switch Dataset' tab.")
        st.stop()
    try:
        df = pd.read_excel(file, sheet_name=sheet)
    except Exception as e:
        st.error(f"Error loading '{file}' → '{sheet}': {e}")
        st.stop()

    # ---- Analytics code ----
    def score_row(y_true, y_pred, name):
        return {"Model": name,
                "Accuracy": round(accuracy_score(y_true, y_pred), 3),
                "Precision": round(precision_score(y_true, y_pred, average='weighted'), 3),
                "Recall": round(recall_score(y_true, y_pred, average='weighted'), 3),
                "F1": round(f1_score(y_true, y_pred, average='weighted'), 3)}

    def prettify_rules(rules_df):
        for c in ("antecedents", "consequents"):
            rules_df[c] = rules_df[c].apply(lambda x: ", ".join(sorted(list(x))))
        return rules_df

    # 1️⃣  DESCRIPTIVE ANALYTICS
    if page == "📊 Descriptive Analytics":
        st.header("📊 Descriptive Portfolio Insights")
        with st.sidebar.expander("Filters", True):
            age_rng = st.slider("Age", int(df.Age.min()), int(df.Age.max()), (int(df.Age.min()), int(df.Age.max())))
            income_rng = st.slider("Annual Income", int(df['Annual Income'].min()), int(df['Annual Income'].max()),
                                   (int(df['Annual Income'].min()), int(df['Annual Income'].max())))
            risk_levels = st.multiselect("Risk Tolerance", df['Risk Tolerance'].unique(), default=list(df['Risk Tolerance'].unique()))
            show_raw = st.checkbox("Show raw data")
        view = df[
            (df.Age.between(*age_rng)) &
            (df['Annual Income'].between(*income_rng)) &
            (df['Risk Tolerance'].isin(risk_levels))
        ]
        st.success(f"Filtered records: {len(view)}")
        if show_raw:
            st.dataframe(view.head())

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Income distribution")
            fig, ax = plt.subplots()
            sns.histplot(view["Annual Income"], kde=True, ax=ax)
            st.pyplot(fig)
        with c2:
            st.subheader("Net worth distribution")
            fig2, ax2 = plt.subplots()
            sns.histplot(view["Net worth"], kde=True, ax=ax2)
            st.pyplot(fig2)

        c3, c4 = st.columns(2)
        with c3:
            st.subheader("Portfolio allocation: Equity (%) vs Age")
            fig3 = px.scatter(view, x="Age", y="Portfolio Equity(%)", color="Risk Tolerance", opacity=0.6)
            st.plotly_chart(fig3, use_container_width=True)
        with c4:
            st.subheader("Recommended Portfolio Counts")
            fig4 = px.histogram(view, x="Recommended Portfolio", color="Risk Tolerance", barmode="group")
            st.plotly_chart(fig4, use_container_width=True)

        c5, c6 = st.columns(2)
        with c5:
            st.subheader("Portfolio Volatility by Portfolio Type")
            fig5 = px.box(view, x="Recommended Portfolio", y="Portfolio Volatility")
            st.plotly_chart(fig5, use_container_width=True)
        with c6:
            st.subheader("Historical Return (%) by Portfolio")
            fig6 = px.box(view, x="Recommended Portfolio", y="Historical Return (%)")
            st.plotly_chart(fig6, use_container_width=True)

    # 2️⃣  CLASSIFICATION
    elif page == "🤖 Classification":
        st.header("🤖 Recommended Portfolio Classifier")
        y = df["Recommended Portfolio"]
        X = pd.get_dummies(df.drop(columns=["UserID", "Recommended Portfolio"]), drop_first=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)
        scaler = StandardScaler().fit(X_train)
        X_train_sc, X_test_sc = scaler.transform(X_train), scaler.transform(X_test)

        models = {
            "KNN":               KNeighborsClassifier(n_neighbors=7),
            "Decision Tree":     DecisionTreeClassifier(max_depth=6, random_state=42),
            "Random Forest":     RandomForestClassifier(n_estimators=300, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42)
        }

        scores, probas = [], {}
        for name, mdl in models.items():
            if name == "KNN":
                mdl.fit(X_train_sc, y_train)
                preds = mdl.predict(X_test_sc)
                probas[name] = mdl.predict_proba(X_test_sc)
            else:
                mdl.fit(X_train, y_train)
                preds = mdl.predict(X_test)
                probas[name] = mdl.predict_proba(X_test)
            scores.append(score_row(y_test, preds, name))

        st.subheader("Metrics")
        st.dataframe(pd.DataFrame(scores).set_index("Model"))

        choice = st.selectbox("Show confusion matrix for:", [s["Model"] for s in scores])
        sel_model = models[choice]
        y_pred = sel_model.predict(X_test_sc if choice == "KNN" else X_test)
        cm = confusion_matrix(y_test, y_pred, labels=y.unique())
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=y.unique(), yticklabels=y.unique(), ax=ax_cm)
        st.pyplot(fig_cm)

        st.subheader("ROC curves (One-vs-Rest, for each class)")
        y_bin = label_binarize(y_test, classes=y.unique())
        fig_roc, ax_roc = plt.subplots()
        for i, name in enumerate(models.keys()):
            if hasattr(models[name], "predict_proba"):
                prob = probas[name]
                for j, class_lbl in enumerate(y.unique()):
                    if prob.shape[1] > 1:
                        fpr, tpr, _ = roc_curve(y_bin[:, j], prob[:, j])
                        ax_roc.plot(fpr, tpr, label=f"{name} - {class_lbl} (AUC={auc(fpr, tpr):.2f})")
        ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray");  ax_roc.legend()
        st.pyplot(fig_roc)

    # 3️⃣  CLUSTERING (K-PROTOTYPES)
    elif page == "🎯 Clustering (K-Prototypes)":
        st.header("🎯 K-Prototypes Portfolio Segmentation")
        num_cols = ["Age", "Investment Horizon", "Annual Income", "Net worth", "Projected ROI 5years",
                    "Portfolio Equity(%)", "Portfolio Bonds(%)", "Portfolio Cash(%)", 
                    "Portfolio RealEstate(%)", "Portfolio Crypto(%)", "Historical Return (%)", "Portfolio Volatility"]
        cat_cols = [c for c in df.columns if c not in num_cols + ["UserID", "Cluster"]]

        df_clean = df.copy()
        df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].mean())
        for c in cat_cols:
            df_clean[c] = df_clean[c].fillna("Missing").astype(str)

        scaler   = StandardScaler()
        X_num    = scaler.fit_transform(df_clean[num_cols])
        X_cat    = df_clean[cat_cols].to_numpy()
        X_mix    = np.hstack([X_num, X_cat])
        cat_idx  = list(range(X_num.shape[1], X_mix.shape[1]))

        k  = st.slider("k (clusters)", 2, 10, 4)
        g  = st.number_input("γ (numeric-vs-categorical weight – 0 = auto)", 0.0, 10.0, 0.0, 0.1)
        γ  = None if g == 0 else g

        kp = KPrototypes(n_clusters=k, init="Huang", n_init=10, gamma=γ, random_state=42, verbose=0)
        clusters = kp.fit_predict(X_mix, categorical=cat_idx)
        df["Cluster"] = clusters
        st.success(f"Clustering complete → {k} segments")

        costs = []
        for ki in range(2, 11):
            km = KPrototypes(n_clusters=ki, n_init=5, random_state=42)
            km.fit_predict(X_mix, categorical=cat_idx)
            costs.append(km.cost_)
        fig_cost, ax_cost = plt.subplots()
        ax_cost.plot(range(2, 11), costs, marker="o")
        ax_cost.set(xlabel="k", ylabel="Cost", title="Cost curve")
        st.pyplot(fig_cost)

        persona_num = df.groupby("Cluster")[num_cols].mean().round(1)
        persona_cat = df.groupby("Cluster")[cat_cols].agg(lambda s: s.mode().iloc[0])
        persona = pd.concat([persona_num, persona_cat], axis=1)
        st.subheader("Cluster personas")
        st.dataframe(persona)

        st.download_button("Download labelled data",
                        df.to_csv(index=False).encode("utf-8"),
                        "clustered_data.csv",
                        "text/csv")

    # 4️⃣  ASSOCIATION RULES
    elif page == "🛒 Association Rules":
        st.header("🛒 Portfolio Allocation Associations (Apriori)")
        alloc_cols = ["Portfolio Equity(%)", "Portfolio Bonds(%)", "Portfolio Cash(%)",
                    "Portfolio RealEstate(%)", "Portfolio Crypto(%)"]
        basket = (df[alloc_cols] > 10).astype(int)
        min_sup  = st.slider("Min support",     0.01, 0.5, 0.05, 0.01)
        min_conf = st.slider("Min confidence",  0.1,  0.9, 0.6,  0.05)
        min_lift = st.slider("Min lift",        1.0,  5.0, 1.2,  0.1)

        if st.button("Run Apriori"):
            freq = apriori(basket, min_support=min_sup, use_colnames=True)
            if freq.empty:
                st.warning("No itemsets — lower support.")
            else:
                rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
                rules = rules[rules["lift"] >= min_lift]
                if rules.empty:
                    st.warning("No rules at these thresholds.")
                else:
                    rules = prettify_rules(rules).sort_values("lift", ascending=False).head(10)
                    st.dataframe(rules[["antecedents", "consequents", "support", "confidence", "lift"]]
                                .style.format({"support":"{:.3f}", "confidence":"{:.2f}", "lift":"{:.2f}"}))

    # 5️⃣  REGRESSION
    else:
        st.header("📈 Regression – Predict Historical Return (%) / Volatility")
        target = st.selectbox("Choose target variable", ["Historical Return (%)", "Portfolio Volatility"])
        y = df[target]
        X = pd.get_dummies(df.drop(columns=["UserID", "Recommended Portfolio", "Historical Return (%)", "Portfolio Volatility"]), drop_first=True)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        regs = {"Linear": LinearRegression(),
                "Ridge":  Ridge(alpha=1.0),
                "Lasso":  Lasso(alpha=0.001),
                "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42)}
        out = []
        for name, r in regs.items():
            r.fit(X_tr, y_tr);  preds = r.predict(X_te)
            out.append({"Model": name,
                        "R2":   round(r.score(X_te, y_te), 3),
                        "RMSE": int(np.sqrt(((y_te - preds) ** 2).mean())),
                        "MAE":  int(np.abs(y_te - preds).mean())})
        st.dataframe(pd.DataFrame(out).set_index("Model"))
