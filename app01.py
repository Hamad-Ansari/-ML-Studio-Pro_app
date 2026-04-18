# ============================================================
#   🚀 AutoML Studio — Production Streamlit ML Application
#   Author  : Senior Python / ML / Streamlit Developer
#   Version : 1.0.0
# ============================================================

import io
import pickle
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 1.  PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoML Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────
# 2.  CUSTOM CSS  (Glassmorphism · Gradients · Animations)
# ─────────────────────────────────────────────────────────────
def inject_css() -> None:
    st.markdown(
        """
        <style>
        /* ── Google Font ── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

        /* ── Global Reset ── */
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        /* ── App Background ── */
        .stApp {
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            min-height: 100vh;
        }

        /* ── Sidebar ── */
        section[data-testid="stSidebar"] {
            background: rgba(255,255,255,0.05);
            backdrop-filter: blur(20px);
            border-right: 1px solid rgba(255,255,255,0.1);
        }
        section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

        /* ── Hero Banner ── */
        .hero-banner {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            border-radius: 20px;
            padding: 3rem 2.5rem;
            text-align: center;
            box-shadow: 0 20px 60px rgba(102,126,234,0.4);
            margin-bottom: 2rem;
            animation: fadeInDown 0.8s ease-out;
        }
        .hero-banner h1 {
            font-size: 3.2rem;
            font-weight: 800;
            color: #ffffff;
            margin: 0;
            text-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        .hero-banner p {
            font-size: 1.15rem;
            color: rgba(255,255,255,0.88);
            margin-top: 0.8rem;
        }

        /* ── Glass Card ── */
        .glass-card {
            background: rgba(255,255,255,0.07);
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 16px;
            padding: 1.8rem;
            backdrop-filter: blur(12px);
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            margin-bottom: 1.5rem;
            animation: fadeIn 0.6s ease-out;
        }

        /* ── Metric Card ── */
        .metric-card {
            background: linear-gradient(135deg, rgba(102,126,234,0.25), rgba(118,75,162,0.25));
            border: 1px solid rgba(102,126,234,0.4);
            border-radius: 14px;
            padding: 1.4rem 1.2rem;
            text-align: center;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .metric-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 30px rgba(102,126,234,0.35);
        }
        .metric-card .metric-value {
            font-size: 2.1rem;
            font-weight: 700;
            color: #a78bfa;
        }
        .metric-card .metric-label {
            font-size: 0.85rem;
            color: rgba(255,255,255,0.65);
            margin-top: 0.3rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* ── Best Model Badge ── */
        .best-model-card {
            background: linear-gradient(135deg, #f6d365, #fda085);
            border-radius: 16px;
            padding: 1.8rem;
            text-align: center;
            box-shadow: 0 10px 40px rgba(253,160,133,0.45);
            animation: pulse 2s infinite;
        }
        .best-model-card h2 { color: #1a1a2e; font-weight: 800; margin: 0; }
        .best-model-card p  { color: rgba(26,26,46,0.75); margin-top: 0.5rem; }

        /* ── Section Title ── */
        .section-title {
            font-size: 1.5rem;
            font-weight: 700;
            color: #c4b5fd;
            border-left: 4px solid #7c3aed;
            padding-left: 0.8rem;
            margin-bottom: 1.2rem;
        }

        /* ── Streamlit Buttons ── */
        div.stButton > button {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 2rem;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102,126,234,0.4);
        }
        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102,126,234,0.6);
            background: linear-gradient(135deg, #764ba2, #667eea);
        }

        /* ── Streamlit Tabs ── */
        .stTabs [data-baseweb="tab-list"] {
            background: rgba(255,255,255,0.06);
            border-radius: 12px;
            padding: 4px;
            gap: 4px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 9px;
            color: rgba(255,255,255,0.6);
            font-weight: 600;
            padding: 0.5rem 1.2rem;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea, #764ba2) !important;
            color: white !important;
        }

        /* ── DataFrame ── */
        .stDataFrame { border-radius: 12px; overflow: hidden; }

        /* ── Alert Boxes ── */
        .stSuccess, .stError, .stWarning, .stInfo {
            border-radius: 12px;
        }

        /* ── Animations ── */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(16px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeInDown {
            from { opacity: 0; transform: translateY(-30px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulse {
            0%, 100% { box-shadow: 0 10px 40px rgba(253,160,133,0.45); }
            50%       { box-shadow: 0 10px 60px rgba(253,160,133,0.75); }
        }

        /* ── Scrollbar ── */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #7c3aed; border-radius: 10px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# 3.  UTILITY / HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def render_metric_card(label: str, value: str) -> str:
    """Return an HTML metric-card string."""
    return f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """


@st.cache_data(show_spinner=False)
def load_example_dataset(name: str) -> pd.DataFrame:
    """Load a seaborn example dataset (cached)."""
    return sns.load_dataset(name)


@st.cache_data(show_spinner=False)
def read_uploaded_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Parse CSV / XLSX / TSV upload (cached by content)."""
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "csv":
        return pd.read_csv(io.BytesIO(file_bytes))
    elif ext in ("xls", "xlsx"):
        return pd.read_excel(io.BytesIO(file_bytes))
    elif ext == "tsv":
        return pd.read_csv(io.BytesIO(file_bytes), sep="\t")
    else:
        st.error("❌ Unsupported file format.")
        st.stop()


def preprocess(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> tuple[np.ndarray, np.ndarray, StandardScaler, dict, list[str]]:
    """
    Full preprocessing pipeline:
      1. Separate features & target
      2. Encode categoricals (LabelEncoder per column)
      3. Impute missing values (IterativeImputer)
      4. Scale features (StandardScaler)
    Returns X_scaled, y, scaler, encoders_dict, categorical_feature_names
    """
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # ── Encode target if categorical ──
    target_encoder = None
    if y.dtype == object or str(y.dtype) == "category":
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y.astype(str))

    # ── Encode categorical features ──
    encoders: dict = {}
    cat_cols: list[str] = []
    for col in X.columns:
        if X[col].dtype == object or str(X[col].dtype) == "category":
            le = LabelEncoder()
            X[col] = X[col].astype(str)
            # Handle NaN before encoding
            X[col] = X[col].fillna("__missing__")
            X[col] = le.fit_transform(X[col])
            encoders[col] = le
            cat_cols.append(col)

    # ── Impute missing values ──
    imputer = IterativeImputer(random_state=42, max_iter=10)
    X_imputed = imputer.fit_transform(X)

    # ── Scale features ──
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled, np.array(y), scaler, encoders, cat_cols, target_encoder


def get_models(problem_type: str) -> dict:
    """Return a dict of model_name → sklearn estimator."""
    if problem_type == "Regression":
        return {
            "Linear Regression":          LinearRegression(),
            "Decision Tree Regressor":    DecisionTreeRegressor(random_state=42),
            "Random Forest Regressor":    RandomForestRegressor(n_estimators=100, random_state=42),
            "Support Vector Regressor":   SVR(),
        }
    else:
        return {
            "Logistic Regression":        LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree Classifier":   DecisionTreeClassifier(random_state=42),
            "Random Forest Classifier":   RandomForestClassifier(n_estimators=100, random_state=42),
            "Support Vector Classifier":  SVC(probability=True, random_state=42),
        }


@st.cache_resource(show_spinner=False)
def train_all_models(
    _X_train: np.ndarray,
    _y_train: np.ndarray,
    problem_type: str,
) -> dict:
    """Train all models and return fitted estimators (cached)."""
    models = get_models(problem_type)
    trained: dict = {}
    for name, model in models.items():
        model.fit(_X_train, _y_train)
        trained[name] = model
    return trained


def evaluate_regression(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    mse   = mean_squared_error(y_test, preds)
    return {
        "MSE":    round(mse, 4),
        "RMSE":   round(np.sqrt(mse), 4),
        "MAE":    round(mean_absolute_error(y_test, preds), 4),
        "R²":     round(r2_score(y_test, preds), 4),
    }


def evaluate_classification(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    return {
        "Accuracy":  round(accuracy_score(y_test, preds), 4),
        "Precision": round(precision_score(y_test, preds, average="weighted", zero_division=0), 4),
        "Recall":    round(recall_score(y_test, preds, average="weighted", zero_division=0), 4),
        "F1 Score":  round(f1_score(y_test, preds, average="weighted", zero_division=0), 4),
    }


def find_best_model(results: dict, problem_type: str) -> str:
    """Return the name of the best-performing model."""
    if problem_type == "Regression":
        return max(results, key=lambda m: results[m]["R²"])
    else:
        return max(results, key=lambda m: results[m]["F1 Score"])


def serialize_model(model) -> bytes:
    """Pickle a model into bytes for download."""
    buf = io.BytesIO()
    pickle.dump(model, buf)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────
# 4.  SECTION RENDERERS
# ─────────────────────────────────────────────────────────────

def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-banner">
            <h1>🤖 AutoML Studio</h1>
            <p>
                Upload your data · Configure your pipeline · Train multiple models · 
                Compare results · Export & Predict — all in one place.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_data_overview(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">📊 Data Overview</div>', unsafe_allow_html=True)

    # ── Top-level metrics ──
    c1, c2, c3, c4 = st.columns(4)
    for col, label, value in zip(
        [c1, c2, c3, c4],
        ["Rows", "Columns", "Missing Values", "Duplicate Rows"],
        [
            f"{df.shape[0]:,}",
            str(df.shape[1]),
            str(df.isnull().sum().sum()),
            str(df.duplicated().sum()),
        ],
    ):
        col.markdown(render_metric_card(label, value), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Preview", "📋 Column Info", "📈 Statistics", "🗺️ Correlation"])

    with tab1:
        st.dataframe(df.head(20), use_container_width=True)

    with tab2:
        info_df = pd.DataFrame({
            "Column":    df.columns.tolist(),
            "Dtype":     [str(d) for d in df.dtypes],
            "Non-Null":  df.notnull().sum().values,
            "Null":      df.isnull().sum().values,
            "Null %":    (df.isnull().mean() * 100).round(2).astype(str) + "%",
            "Unique":    df.nunique().values,
        })
        st.dataframe(info_df, use_container_width=True)

    with tab3:
        st.dataframe(df.describe(include="all").T, use_container_width=True)

    with tab4:
        num_df = df.select_dtypes(include=np.number)
        if num_df.shape[1] >= 2:
            fig = px.imshow(
                num_df.corr(),
                color_continuous_scale="Viridis",
                title="Feature Correlation Matrix",
                aspect="auto",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ Need at least 2 numeric columns for a correlation matrix.")


def render_results(
    results: dict,
    problem_type: str,
    best_name: str,
    trained_models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    st.markdown('<div class="section-title">📈 Evaluation Results</div>', unsafe_allow_html=True)

    # ── Results Table ──
    results_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})

    # Highlight best row
    def highlight_best(row):
        color = "background: rgba(167,139,250,0.25); color: #f0e6ff; font-weight:700;"
        return [color if row["Model"] == best_name else "" for _ in row]

    st.dataframe(
        results_df.style.apply(highlight_best, axis=1),
        use_container_width=True,
    )

    # ── Bar Chart Comparison ──
    metric_key = "R²" if problem_type == "Regression" else "F1 Score"
    fig_bar = px.bar(
        results_df,
        x="Model",
        y=metric_key,
        color=metric_key,
        color_continuous_scale="Purples",
        title=f"Model Comparison — {metric_key}",
        text=metric_key,
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Confusion Matrix (Classification only) ──
    if problem_type == "Classification":
        st.markdown("#### 🔲 Confusion Matrices")
        cols = st.columns(len(trained_models))
        for idx, (name, model) in enumerate(trained_models.items()):
            preds = model.predict(X_test)
            cm = confusion_matrix(y_test, preds)
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                color_continuous_scale="Purples",
                title=name,
                aspect="auto",
            )
            fig_cm.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                margin=dict(t=40, b=10, l=10, r=10),
                coloraxis_showscale=False,
            )
            cols[idx].plotly_chart(fig_cm, use_container_width=True)

    # ── Regression: Actual vs Predicted ──
    if problem_type == "Regression":
        st.markdown("#### 📉 Actual vs Predicted")
        cols = st.columns(len(trained_models))
        for idx, (name, model) in enumerate(trained_models.items()):
            preds = model.predict(X_test)
            fig_sc = px.scatter(
                x=y_test,
                y=preds,
                labels={"x": "Actual", "y": "Predicted"},
                title=name,
                opacity=0.7,
                trendline="ols",
                trendline_color_override="#f093fb",
            )
            fig_sc.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                margin=dict(t=40, b=10, l=10, r=10),
            )
            cols[idx].plotly_chart(fig_sc, use_container_width=True)


def render_best_model_card(best_name: str, results: dict, problem_type: str) -> None:
    metric_key = "R²" if problem_type == "Regression" else "F1 Score"
    score = results[best_name][metric_key]
    st.markdown(
        f"""
        <div class="best-model-card">
            <h2>🏆 Best Model: {best_name}</h2>
            <p>{metric_key} = <strong>{score}</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)


def render_prediction_module(
    best_model,
    feature_cols: list[str],
    scaler: StandardScaler,
    encoders: dict,
    cat_cols: list[str],
    df: pd.DataFrame,
    target_encoder,
) -> None:
    st.markdown('<div class="section-title">🔮 Prediction Module</div>', unsafe_allow_html=True)

    pred_mode = st.radio(
        "Input method",
        ["✍️ Manual Input", "📂 Upload File"],
        horizontal=True,
    )

    if pred_mode == "✍️ Manual Input":
        st.markdown("#### Fill in feature values:")
        input_dict: dict = {}
        cols = st.columns(min(3, len(feature_cols)))
        for i, col in enumerate(feature_cols):
            with cols[i % 3]:
                if col in cat_cols:
                    le = encoders[col]
                    options = [c for c in le.classes_ if c != "__missing__"]
                    chosen = st.selectbox(f"🏷️ {col}", options, key=f"pred_{col}")
                    input_dict[col] = le.transform([chosen])[0]
                else:
                    col_min  = float(df[col].min(skipna=True))
                    col_max  = float(df[col].max(skipna=True))
                    col_mean = float(df[col].mean(skipna=True))
                    val = st.slider(
                        f"🔢 {col}",
                        min_value=col_min,
                        max_value=col_max,
                        value=col_mean,
                        key=f"pred_{col}",
                    )
                    input_dict[col] = val

        if st.button("🔮 Predict"):
            row = np.array([[input_dict[c] for c in feature_cols]])
            row_scaled = scaler.transform(row)
            pred = best_model.predict(row_scaled)

            # Inverse-transform target if encoded
            if target_encoder is not None:
                pred_label = target_encoder.inverse_transform(pred.astype(int))[0]
            else:
                pred_label = round(float(pred[0]), 4)

            st.markdown(
                f"""
                <div class="glass-card" style="text-align:center;">
                    <div style="font-size:1.1rem; color:rgba(255,255,255,0.6);">Prediction Result</div>
                    <div style="font-size:3rem; font-weight:800; color:#a78bfa; margin-top:0.5rem;">
                        {pred_label}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    else:  # File upload
        pred_file = st.file_uploader(
            "Upload prediction data (same feature columns, no target)",
            type=["csv", "xlsx", "tsv"],
            key="pred_upload",
        )
        if pred_file:
            pred_df = read_uploaded_file(pred_file.read(), pred_file.name)
            st.dataframe(pred_df.head(), use_container_width=True)

            if st.button("🔮 Run Predictions"):
                try:
                    X_new = pred_df[feature_cols].copy()
                    for col in cat_cols:
                        if col in X_new.columns:
                            X_new[col] = X_new[col].astype(str).fillna("__missing__")
                            X_new[col] = encoders[col].transform(X_new[col])

                    imp   = IterativeImputer(random_state=42, max_iter=10)
                    X_imp = imp.fit_transform(X_new)
                    X_sc  = scaler.transform(X_imp)
                    preds = best_model.predict(X_sc)

                    if target_encoder is not None:
                        preds = target_encoder.inverse_transform(preds.astype(int))

                    pred_df["🎯 Prediction"] = preds
                    st.success("✅ Predictions generated successfully!")
                    st.dataframe(pred_df, use_container_width=True)

                    csv_out = pred_df.to_csv(index=False).encode()
                    st.download_button(
                        "⬇️ Download Predictions CSV",
                        data=csv_out,
                        file_name="predictions.csv",
                        mime="text/csv",
                    )
                except Exception as exc:
                    st.error(f"❌ Error during prediction: {exc}")


# ─────────────────────────────────────────────────────────────
# 5.  SIDEBAR
# ─────────────────────────────────────────────────────────────

def render_sidebar() -> tuple:
    """
    Render sidebar controls.
    Returns (df, problem_type, feature_cols, target_col, test_size, selected_models).
    Returns None values if configuration is incomplete.
    """
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown("---")

        # ── Data Source ──
        st.markdown("### 📂 Data Source")
        data_source = st.radio(
            "Choose data source",
            ["📁 Upload Dataset", "📚 Example Dataset"],
            label_visibility="collapsed",
        )

        df = None

        if data_source == "📁 Upload Dataset":
            uploaded = st.file_uploader(
                "Upload CSV / XLSX / TSV",
                type=["csv", "xlsx", "tsv", "xls"],
            )
            if uploaded:
                with st.spinner("Reading file…"):
                    df = read_uploaded_file(uploaded.read(), uploaded.name)
                st.success(f"✅ Loaded: {uploaded.name}")

        else:
            example_name = st.selectbox(
                "Choose example dataset",
                ["iris", "titanic", "tips"],
            )
            with st.spinner("Loading dataset…"):
                df = load_example_dataset(example_name)
            st.success(f"✅ Loaded: {example_name}")

        if df is None:
            return None, None, None, None, None, None

        st.markdown("---")

        # ── Column Selection ──
        st.markdown("### 🎯 Target & Features")
        all_cols = df.columns.tolist()
        target_col = st.selectbox("Target column", all_cols, index=len(all_cols) - 1)
        feature_cols = st.multiselect(
            "Feature columns",
            [c for c in all_cols if c != target_col],
            default=[c for c in all_cols if c != target_col],
        )

        # ── Problem Type ──
        st.markdown("---")
        st.markdown("### 🤖 Problem Type")
        problem_type = st.radio(
            "Select problem type",
            ["Classification", "Regression"],
            label_visibility="collapsed",
        )

        # ── Train-Test Split ──
        st.markdown("---")
        st.markdown("### ✂️ Train / Test Split")
        test_size = st.slider(
            "Test set size",
            min_value=0.10,
            max_value=0.40,
            value=0.20,
            step=0.05,
            format="%.0f%%",
            help="Fraction of data reserved for testing",
        )
        st.caption(
            f"Train: {int((1 - test_size) * 100)}%  |  Test: {int(test_size * 100)}%"
        )

        # ── Model Selection ──
        st.markdown("---")
        st.markdown("### 🔬 Models to Train")
        model_options = list(get_models(problem_type).keys())
        selected_models = st.multiselect(
            "Choose models",
            model_options,
            default=model_options,
        )

        st.markdown("---")
        st.markdown(
            "<div style='text-align:center; color:rgba(255,255,255,0.3); font-size:0.75rem;'>"
            "AutoML Studio v1.0 · Built with ❤️ + Streamlit"
            "</div>",
            unsafe_allow_html=True,
        )

    return df, problem_type, feature_cols, target_col, test_size, selected_models


# ─────────────────────────────────────────────────────────────
# 6.  MAIN APPLICATION
# ─────────────────────────────────────────────────────────────

def main() -> None:
    inject_css()
    render_hero()

    # ── Get sidebar config ──
    df, problem_type, feature_cols, target_col, test_size, selected_models = render_sidebar()

    if df is None:
        st.markdown(
            """
            <div class="glass-card" style="text-align:center; padding:3rem;">
                <div style="font-size:4rem;">👈</div>
                <div style="color:rgba(255,255,255,0.7); font-size:1.2rem; margin-top:1rem;">
                    Select a data source from the sidebar to get started.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # ── Validation ──
    if not feature_cols:
        st.warning("⚠️ Please select at least one feature column in the sidebar.")
        return

    if not selected_models:
        st.warning("⚠️ Please select at least one model in the sidebar.")
        return

    # ── Data Overview ──
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        render_data_overview(df)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Run Analysis Button ──
    st.markdown("<br>", unsafe_allow_html=True)
    col_btn, col_info = st.columns([1, 3])
    run_analysis = col_btn.button("🚀 Run Analysis", use_container_width=True)
    col_info.info(
        "ℹ️ Click **Run Analysis** to start preprocessing and model training."
    )

    if not run_analysis:
        return

    # ════════════════════════════════════════════════════════
    # PIPELINE STARTS HERE
    # ════════════════════════════════════════════════════════
    main_tab, train_tab, result_tab, pred_tab = st.tabs(
        ["🗂️ Pipeline Info", "🏋️ Training", "📊 Results", "🔮 Predictions"]
    )

    # ── TAB 1: Preprocessing Info ──
    with main_tab:
        st.markdown('<div class="section-title">⚙️ Preprocessing Pipeline</div>', unsafe_allow_html=True)

        with st.spinner("🔄 Preprocessing data…"):
            X, y, scaler, encoders, cat_cols, target_encoder = preprocess(
                df, feature_cols, target_col
            )
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

        c1, c2, c3, c4 = st.columns(4)
        for col_el, label, value in zip(
            [c1, c2, c3, c4],
            ["Train Samples", "Test Samples", "Features", "Encoded Cols"],
            [str(X_train.shape[0]), str(X_test.shape[0]), str(X.shape[1]), str(len(cat_cols))],
        ):
            col_el.markdown(render_metric_card(label, value), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        steps = {
            "🧹 Missing Values":    "IterativeImputer  (max_iter=10)",
            "🔠 Categorical Cols":  f"{cat_cols if cat_cols else 'None detected'}  →  LabelEncoder",
            "📏 Feature Scaling":   "StandardScaler (zero-mean, unit-variance)",
            "✂️ Train/Test Split":  f"{int((1-test_size)*100)}% / {int(test_size*100)}%  (random_state=42)",
        }
        for step, detail in steps.items():
            st.markdown(
                f"""
                <div class="glass-card" style="padding:1rem 1.4rem; margin-bottom:0.6rem;">
                    <span style="font-weight:700; color:#c4b5fd;">{step}</span>
                    <span style="color:rgba(255,255,255,0.65); margin-left:1rem;">{detail}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.success("✅ Preprocessing complete! Switch to the **Training** tab.")

    # ── TAB 2: Training ──
    with train_tab:
        st.markdown('<div class="section-title">🏋️ Model Training</div>', unsafe_allow_html=True)

        train_btn = st.button("▶️ Train Models", use_container_width=False)

        if train_btn:
            progress_bar = st.progress(0, text="Initialising…")
            trained_models: dict = {}
            models_to_train = {
                k: v for k, v in get_models(problem_type).items() if k in selected_models
            }

            for idx, (name, model) in enumerate(models_to_train.items()):
                progress_bar.progress(
                    int((idx / len(models_to_train)) * 100),
                    text=f"⚙️ Training: {name}",
                )
                with st.spinner(f"Training {name}…"):
                    model.fit(X_train, y_train)
                    trained_models[name] = model

            progress_bar.progress(100, text="✅ All models trained!")
            st.session_state["trained_models"]  = trained_models
            st.session_state["X_test"]          = X_test
            st.session_state["y_test"]          = y_test
            st.session_state["scaler"]          = scaler
            st.session_state["encoders"]        = encoders
            st.session_state["cat_cols"]        = cat_cols
            st.session_state["target_encoder"]  = target_encoder
            st.session_state["problem_type"]    = problem_type
            st.session_state["feature_cols"]    = feature_cols
            st.session_state["df"]              = df
            st.success("✅ Training complete! Switch to the **Results** tab.")

        elif "trained_models" not in st.session_state:
            st.info("ℹ️ Click **Train Models** to begin.")

    # ── TAB 3: Results ──
    with result_tab:
        if "trained_models" not in st.session_state:
            st.info("ℹ️ Train the models first (Training tab).")
        else:
            trained_models = st.session_state["trained_models"]
            X_test_s       = st.session_state["X_test"]
            y_test_s       = st.session_state["y_test"]
            pt             = st.session_state["problem_type"]

            # Evaluate
            results: dict = {}
            for name, model in trained_models.items():
                if pt == "Regression":
                    results[name] = evaluate_regression(model, X_test_s, y_test_s)
                else:
                    results[name] = evaluate_classification(model, X_test_s, y_test_s)

            best_name = find_best_model(results, pt)

            # Best model banner
            render_best_model_card(best_name, results, pt)

            # Detailed results
            render_results(results, pt, best_name, trained_models, X_test_s, y_test_s)

            # ── Export ──
            st.markdown("---")
            st.markdown('<div class="section-title">💾 Export Best Model</div>', unsafe_allow_html=True)
            if st.checkbox("📦 Download the best model as `.pkl`?"):
                model_bytes = serialize_model(trained_models[best_name])
                st.download_button(
                    label=f"⬇️ Download  {best_name}.pkl",
                    data=model_bytes,
                    file_name=f"{best_name.replace(' ', '_')}.pkl",
                    mime="application/octet-stream",
                )

    # ── TAB 4: Predictions ──
    with pred_tab:
        if "trained_models" not in st.session_state:
            st.info("ℹ️ Train the models first (Training tab).")
        else:
            if st.checkbox("🔮 Make predictions using the best model?"):
                # Re-evaluate to get best model
                trained_models  = st.session_state["trained_models"]
                X_test_s        = st.session_state["X_test"]
                y_test_s        = st.session_state["y_test"]
                pt              = st.session_state["problem_type"]
                scaler_s        = st.session_state["scaler"]
                encoders_s      = st.session_state["encoders"]
                cat_cols_s      = st.session_state["cat_cols"]
                feature_cols_s  = st.session_state["feature_cols"]
                df_s            = st.session_state["df"]
                target_enc_s    = st.session_state["target_encoder"]

                results_for_best: dict = {}
                for name, model in trained_models.items():
                    if pt == "Regression":
                        results_for_best[name] = evaluate_regression(model, X_test_s, y_test_s)
                    else:
                        results_for_best[name] = evaluate_classification(model, X_test_s, y_test_s)

                best_name   = find_best_model(results_for_best, pt)
                best_model  = trained_models[best_name]

                render_prediction_module(
                    best_model,
                    feature_cols_s,
                    scaler_s,
                    encoders_s,
                    cat_cols_s,
                    df_s,
                    target_enc_s,
                )


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()