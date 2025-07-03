# app.py – Walmart Analytics Dashboard (with per-section error handling)
# ---------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import traceback

# Optional Plotly support
try:
    import plotly.express as px
    HAS_PLOTLY = True
except ModuleNotFoundError:
    HAS_PLOTLY = False
    import warnings
    warnings.warn("Plotly not installed – falling back to Matplotlib.")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, silhouette_score
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from mlxtend.frequent_patterns import apriori, association_rules

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
FILE_NAME       = "Anirudh' data set.xlsx"
PREFERRED_SHEET = "Dataset (2)"

st.set_page_config(page_title="Walmart Sales Intelligence",
                   page_icon="🛒",
                   layout="wide")

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def metric_row(y, p, name):
    return {"Model": name,
            "Accuracy":  round(accuracy_score(y, p), 3),
            "Precision": round(precision_score(y, p, average="weighted"), 3),
            "Recall":    round(recall_score(y, p, average="weighted"), 3),
            "F1":        round(f1_score(y, p, average="weighted"), 3)}

def tidy_sets(r):
    for c in ("antecedents", "consequents"):
        r[c] = r[c].apply(lambda x: ", ".join(sorted(list(x))))
    return r

def safe_section(label):
    """
    Decorator that turns any exception inside a section into a nice Streamlit error
    without killing the whole app.
    """
    def deco(func):
        def wrapped(*args, **kwargs):
            try:
                func(*args, **kwargs)
            except Exception as e:
                st.error(f"🚨 **{label}** ran into an error:")
                st.code("".join(traceback.format_exception_only(type(e), e)).strip())
                # uncomment next line if you want full trace visible
                # st.exception(e)
        return wrapped
    return deco

# ---------------------------------------------------------------------
# DATA LOADER
# ---------------------------------------------------------------------
@st.cache_data
def load_data(path: str, preferred: str):
    try:
        xl = pd.ExcelFile(path)
    except FileNotFoundError:
        st.error(f"❌ File *{path}* not found.")
        return pd.DataFrame()

    sheet_to_use = preferred if preferred in xl.sheet_names else xl.sheet_names[0]
    if preferred not in xl.sheet_names:
        st.warning(f"Sheet *{preferred}* not found – using *{sheet_to_use}*.")
    else:
        st.info(f"Loaded sheet *{sheet_to_use}*")

    return pd.read_excel(xl, sheet_name=sheet_to_use)

with st.spinner("Loading data…"):
    df = load_data(FILE_NAME, PREFERRED_SHEET)

if df.empty:
    st.stop()

numeric_cols     = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

# ---------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------------------
st.sidebar.title("🏷️ Navigation")
tab = st.sidebar.radio(
    "Choose module",
    ("📊 Descriptive Analytics",
     "🤖 Classification",
     "🎯 Clustering",
     "🛒 Association Rules",
     "📈 Regression")
)

# ---------------------------------------------------------------------
# SECTION IMPLEMENTATIONS
# ---------------------------------------------------------------------
@safe_section("📊 Descriptive Analytics")
def section_descriptive():
    st.header("📊 Descriptive Analytics")
    # ----- filters -----
    with st.sidebar.expander("Filters", True):
        if not numeric_cols:
            st.warning("No numeric columns in data.")
            return
        num_col = st.selectbox("Numeric filter", numeric_cols, 0)
        rng = st.slider(f"{num_col} range",
                        float(df[num_col].min()),
                        float(df[num_col].max()),
                        (float(df[num_col].min()), float(df[num_col].max())))
        cat_filters = {c: st.multiselect(c,
                                         df[c].dropna().unique().tolist(),
                                         default=df[c].dropna().unique().tolist())
                       for c in categorical_cols[:5]}
        show_raw = st.checkbox("Show raw data")

    mask = df[num_col].between(*rng)
    for c, vals in cat_filters.items():
        mask &= df[c].isin(vals)
    dff = df[mask]

    st.success(f"{len(dff):,} rows after filtering")
    if show_raw:
        st.dataframe(dff.head())

    # ----- plots -----
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        ax.hist(dff[num_col].dropna(), bins=30)
        ax.set_xlabel(num_col)
        st.pyplot(fig)

    with c2:
        fig2, ax2 = plt.subplots()
        corr = dff[numeric_cols].corr()
        if corr.isna().all().all():
            st.warning("Correlation matrix could not be computed.")
        else:
            im = ax2.imshow(corr, aspect="auto")
            ax2.set_xticks(range(len(corr))); ax2.set_xticklabels(corr.columns, rotation=90)
            ax2.set_yticks(range(len(corr))); ax2.set_yticklabels(corr.columns)
            fig2.colorbar(im)
            st.pyplot(fig2)

    # scatter with optional linear fit
    if len(numeric_cols) >= 2:
        xcol, ycol = numeric_cols[:2]
        if HAS_PLOTLY:
            fig3 = px.scatter(dff, x=xcol, y=ycol, opacity=0.6, height=400)
            if dff[[xcol, ycol]].dropna().shape[0] > 1:
                m, b = np.polyfit(dff[xcol], dff[ycol], 1)
                xs = np.linspace(dff[xcol].min(), dff[xcol].max(), 200)
                fig3.add_scatter(x=xs, y=m*xs+b, mode="lines",
                                 name="Linear fit", line=dict(dash="dash"))
            st.plotly_chart(fig3, use_container_width=True)
        else:
            fig3, ax3 = plt.subplots()
            ax3.scatter(dff[xcol], dff[ycol], alpha=0.6)
            if dff[[xcol, ycol]].dropna().shape[0] > 1:
                m, b = np.polyfit(dff[xcol], dff[ycol], 1)
                xs = np.linspace(dff[xcol].min(), dff[ycol].max(), 200)
                ax3.plot(xs, m*xs+b, linestyle="--")
            ax3.set_xlabel(xcol); ax3.set_ylabel(ycol)
            st.pyplot(fig3)

@safe_section("🤖 Classification")
def section_classification():
    st.header("🤖 Classification")
    if not categorical_cols:
        st.warning("No categorical columns to predict.")
        return
    target_col = st.selectbox("Pick categorical target", categorical_cols)
    if not target_col:
        return
    y = df[target_col]
    X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)
    if X.empty:
        st.warning("No features after one-hot encoding.")
        return
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                          stratify=y if y.nunique()>1 else None,
                                          random_state=42)
    scaler = StandardScaler().fit(Xtr.select_dtypes(np.number))
    def scale(d):
        d2 = d.copy()
        d2.loc[:, scaler.feature_names_in_] = \
            scaler.transform(d2[scaler.feature_names_in_])
        return d2
    Xtr_sc, Xte_sc = scale(Xtr), scale(Xte)

    models = {
        "KNN":            KNeighborsClassifier(n_neighbors=7),
        "Decision Tree":  DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest":  RandomForestClassifier(n_estimators=300, random_state=42),
        "Gradient Boost": GradientBoostingClassifier(random_state=42),
    }
    scores, probas = [], {}
    for name, mdl in models.items():
        mdl.fit(Xtr_sc if name=="KNN" else Xtr, ytr)
        pred = mdl.predict(Xte_sc if name=="KNN" else Xte)
        scores.append(metric_row(yte, pred, name))
        if y.nunique()==2 and hasattr(mdl, "predict_proba"):
            probas[name] = mdl.predict_proba(Xte_sc if name=="KNN" else Xte)[:,1]

    st.dataframe(pd.DataFrame(scores).set_index("Model"))

    cm_model = st.selectbox("Confusion matrix for", list(models.keys()))
    cm_pred  = models[cm_model].predict(Xte_sc if cm_model=="KNN" else Xte)
    cm       = confusion_matrix(yte, cm_pred)
    fig_cm, ax_cm = plt.subplots()
    ax_cm.imshow(cm, cmap="Blues"); ax_cm.set_xlabel("Pred"); ax_cm.set_ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(j, i, cm[i,j], ha="center", va="center")
    st.pyplot(fig_cm)

    if y.nunique()==2 and probas:
        fig_roc, ax_roc = plt.subplots()
        for name, pr in probas.items():
            fpr, tpr, _ = roc_curve(yte, pr)
            ax_roc.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.2f})")
        ax_roc.plot([0,1],[0,1],"--", color="grey")
        ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR"); ax_roc.legend()
        st.pyplot(fig_roc)

@safe_section("🎯 Clustering")
def section_clustering():
    st.header("🎯 K-Means Clustering")
    if len(numeric_cols) < 2:
        st.warning("Need at least two numeric columns for clustering.")
        return

    k = st.slider("k (clusters)", 2, 10, 4)
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(df[numeric_cols])
    df["Cluster"] = km.labels_

    inertias = [KMeans(n_clusters=i, n_init=10, random_state=42)
                .fit(df[numeric_cols]).inertia_ for i in range(2, 11)]
    sil = [silhouette_score(df[numeric_cols],
                            KMeans(n_clusters=i, n_init=10, random_state=42)
                            .fit_predict(df[numeric_cols])) for i in range(2, 11)]

    cA, cB = st.columns(2)
    with cA:
        fig_el, ax_el = plt.subplots()
        ax_el.plot(range(2,11), inertias, marker="o")
        ax_el.set_xlabel("k"); ax_el.set_ylabel("Inertia"); ax_el.set_title("Elbow")
        st.pyplot(fig_el)
    with cB:
        fig_si, ax_si = plt.subplots()
        ax_si.plot(range(2,11), sil, marker="s")
        ax_si.set_xlabel("k"); ax_si.set_ylabel("Silhouette"); ax_si.set_title("Silhouette")
        st.pyplot(fig_si)

    st.subheader("Cluster centroids")
    st.dataframe(pd.DataFrame(km.cluster_centers_, columns=numeric_cols).round(2))

@safe_section("🛒 Association Rules")
def section_association():
    st.header("🛒 Association Rules")
    bin_cols = [c for c in df.columns
                if df[c].dropna().isin([0,1,True,False]).all()]
    use_cols = st.multiselect("Columns to include", bin_cols+categorical_cols,
                              default=bin_cols[:20] if bin_cols else [])
    if not use_cols:
        st.info("Select at least one column to continue.")
        return

    min_sup  = st.slider("Min support",    0.01, 0.5, 0.05, 0.01)
    min_conf = st.slider("Min confidence", 0.10, 0.9, 0.60, 0.05)
    min_lift = st.slider("Min lift",       1.00, 5.0, 1.20, 0.10)

    if st.button("Run Apriori"):
        basket = pd.get_dummies(df[use_cols].astype(str), prefix=use_cols)
        frequent = apriori(basket.astype(bool), min_support=min_sup, use_colnames=True)
        if frequent.empty:
            st.warning("No frequent itemsets at this support.")
            return
        rules = association_rules(frequent, metric="confidence", min_threshold=min_conf)
        rules = rules[rules["lift"] >= min_lift]
        if rules.empty:
            st.warning("No rules meet confidence/lift thresholds.")
        else:
            st.dataframe(tidy_sets(rules)
                         .sort_values("lift", ascending=False)
                         .head(10)
                         .style.format({"support":"{:.3f}",
                                        "confidence":"{:.2f}",
                                        "lift":"{:.2f}"}))

@safe_section("📈 Regression")
def section_regression():
    st.header("📈 Regression")
    if not numeric_cols:
        st.warning("No numeric columns to model.")
        return
    target = st.selectbox("Numeric target", numeric_cols)
    if not target:
        return
    y = df[target]
    X = pd.get_dummies(df.drop(columns=[target]), drop_first=True)
    if X.empty:
        st.warning("No features after one-hot encoding.")
        return
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    regs = {
        "Linear": LinearRegression(),
        "Ridge":  Ridge(alpha=1.0),
        "Lasso":  Lasso(alpha=0.001),
        "DTReg":  DecisionTreeRegressor(max_depth=6, random_state=42)
    }
    out = []
    for n, r in regs.items():
        r.fit(Xtr, ytr); p = r.predict(Xte)
        out.append({"Model":n,
                    "R²": round(r.score(Xte,yte),3),
                    "RMSE": int(np.sqrt(((yte-p)**2).mean())),
                    "MAE": int(np.abs(yte-p).mean())})
    st.dataframe(pd.DataFrame(out).set_index("Model"))

# ---------------------------------------------------------------------
# ROUTER
# ---------------------------------------------------------------------
if   tab.startswith("📊"): section_descriptive()
elif tab.startswith("🤖"): section_classification()
elif tab.startswith("🎯"): section_clustering()
elif tab.startswith("🛒"): section_association()
else:                      section_regression()
