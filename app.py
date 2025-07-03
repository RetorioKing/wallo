# walmart_dashboard.py  ──  Streamlit analytics suite (Jul-2025)

# ──────────────────── imports & optional dependencies ────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import plotly.express as px
    HAS_PLOTLY = True
except ModuleNotFoundError:
    HAS_PLOTLY = False
    import warnings
    warnings.warn("Plotly not installed – using Matplotlib instead.")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from kmodes.kprototypes import KPrototypes
from mlxtend.frequent_patterns import apriori, association_rules

# ──────────────────── CONFIG – edit if paths/sheet change ─────────────────
DATA_PATH  = "Anirudh' data set.xlsx"   # <── adjust if needed
SHEET_NAME = "Dataset (2)"              # <── adjust if needed

st.set_page_config(page_title="Walmart Analytics Dashboard", page_icon="📦", layout="wide")

# ──────────────────── helpers ─────────────────────────────────────────────
def safe_page(func):
    """Decorator: wrap every page in a try/except so one crash
       doesn’t kill the whole app."""
    def _wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            st.error(f"⛔ {e}")
            st.exception(e)           # expandable traceback
    return _wrapper


@st.cache_data(show_spinner=False)
def load_data(path: str = DATA_PATH, sheet: str = SHEET_NAME) -> pd.DataFrame | None:
    """Read Excel and handle the three most common failure modes."""
    try:
        return pd.read_excel(path, sheet_name=sheet)
    except FileNotFoundError:
        st.error(f"File not found → `{path}`")
    except ValueError as ve:          # wrong sheet name
        st.error(f"Sheet “{sheet}” not found in workbook.")
    except Exception as ex:
        st.error("Failed to read the dataset.")
        st.exception(ex)
    return None


def score_row(y_true, y_pred, name):
    return dict(Model=name,
                Accuracy=round(accuracy_score(y_true, y_pred), 3),
                Precision=round(precision_score(y_true, y_pred), 3),
                Recall=round(recall_score(y_true, y_pred), 3),
                F1=round(f1_score(y_true, y_pred), 3))


def prettify_rules(rules_df):
    for c in ("antecedents", "consequents"):
        rules_df[c] = rules_df[c].apply(lambda x: ", ".join(sorted(list(x))))
    return rules_df


# ──────────────────── DATA LOAD ───────────────────────────────────────────
df = load_data()
if df is None:
    st.stop()   # nothing else makes sense without data

# ──────────────────── SIDEBAR NAV ─────────────────────────────────────────
st.sidebar.title("📦 Walmart Modules")
page = st.sidebar.radio(
    "Choose analytics module",
    ["📊 Descriptive",
     "🤖 Classification",
     "🎯 Clustering",
     "🛒 Association Rules",
     "📈 Regression"]
)

# ──────────────────── 1️⃣ DESCRIPTIVE ─────────────────────────────────────
@safe_page
def page_descriptive():
    st.header("📊 Descriptive Analytics")
    with st.sidebar.expander("Filters", True):
        num_cols = df.select_dtypes("number").columns.tolist()
        cat_cols = df.select_dtypes(exclude="number").columns.tolist()

        filter_num = st.multiselect("Numeric columns for range filter", num_cols)
        filter_cat = st.multiselect("Categorical columns for selection", cat_cols)

        query_parts = []
        for c in filter_num:
            min_v, max_v = st.slider(f"{c} range",
                                     float(df[c].min()), float(df[c].max()),
                                     (float(df[c].min()), float(df[c].max())))
            query_parts.append(f"({c} >= {min_v}) & ({c} <= {max_v})")
        for c in filter_cat:
            opts = st.multiselect(f"{c} values", df[c].unique().tolist(),
                                  default=df[c].unique().tolist())
            query_parts.append(f"`{c}` in @opts")

    view = df.query(" & ".join(query_parts)) if query_parts else df
    st.success(f"Rows after filter : {len(view)}")

    # quick numeric profile
    st.subheader("Numeric summary")
    st.dataframe(view.describe().T.rename_axis("Feature"))

    # histogram for first numeric column
    num_example = view.select_dtypes("number").columns[0]
    st.subheader(f"Histogram – {num_example}")
    if HAS_PLOTLY:
        st.plotly_chart(px.histogram(view, x=num_example), use_container_width=True)
    else:
        fig, ax = plt.subplots()
        sns.histplot(view[num_example], kde=True, ax=ax)
        st.pyplot(fig)


# ──────────────────── 2️⃣ CLASSIFICATION ─────────────────────────────────
@safe_page
def page_classification():
    st.header("🤖 Classification")

    # pick a categorical target
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    target = st.sidebar.selectbox("Target variable", cat_cols)

    y = df[target]
    X = pd.get_dummies(df.drop(columns=[target]), drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.25, random_state=42
    )

    # Scaling numeric features for KNN
    scaler = StandardScaler().fit(X_train)
    X_train_sc = scaler.transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models = {
        "KNN":               KNeighborsClassifier(n_neighbors=7),
        "Decision Tree":     DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest":     RandomForestClassifier(n_estimators=300, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    metrics, probas = [], {}
    for name, mdl in models.items():
        X_tr = X_train_sc if name == "KNN" else X_train
        X_te = X_test_sc  if name == "KNN" else X_test
        mdl.fit(X_tr, y_train)
        preds = mdl.predict(X_te)
        metrics.append(score_row(y_test, preds, name))
        if hasattr(mdl, "predict_proba"):
            probas[name] = mdl.predict_proba(X_te)[:, 1]

    st.subheader("Performance")
    st.dataframe(pd.DataFrame(metrics).set_index("Model"))

    # Confusion matrix for selected model
    sel = st.selectbox("Confusion matrix for", list(models.keys()))
    mdl  = models[sel]
    X_te = X_test_sc if sel == "KNN" else X_test
    cm   = confusion_matrix(y_test, mdl.predict(X_te))

    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
    ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

    # ROC curves (only if binary classification)
    if len(y.unique()) == 2 and probas:
        st.subheader("ROC curves")
        fig, ax = plt.subplots()
        for name, pr in probas.items():
            fpr, tpr, _ = roc_curve(y_test, pr)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.2f})")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.legend(); st.pyplot(fig)


# ──────────────────── 3️⃣ CLUSTERING ─────────────────────────────────────
@safe_page
def page_clustering():
    st.header("🎯 K-Prototypes Clustering")

    num_cols = df.select_dtypes("number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    k   = st.sidebar.slider("Number of clusters (k)", 2, 10, 4)
    gamma = st.sidebar.number_input("Gamma (numeric/cat weight, 0 = auto)",
                                    0.0, 10.0, 0.0, step=0.1)
    γ = None if gamma == 0 else gamma

    # basic missing-value handling
    X_num = df[num_cols].fillna]

